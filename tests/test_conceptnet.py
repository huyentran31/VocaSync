"""
Deterministic test for conceptnet_lookup (Tool #9) — fully MOCKED, no network.

We patch conceptnet_lookup.requests.get with a canned ConceptNet response and assert
the tool's contract:
  • relation mapping  : /r/PartOf->part_of, /r/UsedFor->used_for, /r/HasContext->has_context
  • /r/RelatedTo      : DROPPED (noisiest relation)
  • weight filter     : edges below min_weight removed
  • direction filter  : only edges whose START is the query term are kept
  • language filter    : non-English ends dropped
  • every Edge.source == "conceptnet" and type in EDGE_TYPES

Also checks enrich() folds vetted cn_edges into the node via `keep_edges` (offline, no key).

Offline, no key. Run: python tests/test_conceptnet.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import conceptnet_lookup as cn
from schema import EDGE_TYPES


class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# A realistic-ish ConceptNet payload for /c/en/key
_PAYLOAD = {
    "edges": [
        # kept: UsedFor, START=key, en, weight ok
        {"rel": {"@id": "/r/UsedFor"}, "weight": 2.5,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/open_door", "language": "en", "label": "open door"}},
        # kept: PartOf, START=key
        {"rel": {"@id": "/r/PartOf"}, "weight": 2.0,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/keyboard", "language": "en", "label": "keyboard"}},
        # kept: HasContext (HIGH weight -> should sort first among has_context)
        {"rel": {"@id": "/r/HasContext"}, "weight": 3.0,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/music", "language": "en", "label": "music"}},
        # kept: HasContext (LOWER weight -> should sort AFTER music)
        {"rel": {"@id": "/r/HasContext"}, "weight": 1.6,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/cryptography", "language": "en", "label": "cryptography"}},
        # DROPPED: RelatedTo (noisy relation, unmapped)
        {"rel": {"@id": "/r/RelatedTo"}, "weight": 5.0,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/lock", "language": "en", "label": "lock"}},
        # DROPPED: weight below 1.5
        {"rel": {"@id": "/r/UsedFor"}, "weight": 1.0,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/en/win_game", "language": "en", "label": "win game"}},
        # DROPPED: wrong direction (key is the END, not START)
        {"rel": {"@id": "/r/PartOf"}, "weight": 4.0,
         "start": {"@id": "/c/en/teeth", "language": "en", "label": "teeth"},
         "end": {"@id": "/c/en/key", "language": "en", "label": "key"}},
        # DROPPED: non-English end
        {"rel": {"@id": "/r/UsedFor"}, "weight": 4.0,
         "start": {"@id": "/c/en/key", "language": "en", "label": "key"},
         "end": {"@id": "/c/fr/porte", "language": "fr", "label": "porte"}},
    ]
}


def test_conceptnet_mapping_and_filters(monkeypatch=None):
    cn.requests.get = lambda *a, **k: _FakeResp(_PAYLOAD)  # monkeypatch the network call

    r = cn.conceptnet_lookup("key", min_weight=1.5)
    assert r["found"] is True
    edges = r["edges"]

    pairs = {(e["type"], e["target"]) for e in edges}
    # kept ones
    assert ("used_for", "open door") in pairs
    assert ("part_of", "keyboard") in pairs
    assert ("has_context", "music") in pairs
    # dropped ones
    assert all(e["type"] != "synonym" for e in edges)          # no RelatedTo leaked as anything
    assert ("used_for", "win game") not in pairs               # weight filter
    assert all(e["target"] != "porte" for e in edges)          # language filter
    assert ("part_of", "key") not in pairs                     # direction filter (teeth->key dropped)
    assert not any(e["target"] == "lock" for e in edges)       # RelatedTo dropped

    assert all(e["source"] == "conceptnet" for e in edges)
    assert all(e["type"] in EDGE_TYPES for e in edges)

    # weight sort: among has_context edges, the higher-weight 'music' must come before 'cryptography'
    hc = [e["target"] for e in edges if e["type"] == "has_context"]
    assert hc.index("music") < hc.index("cryptography"), f"has_context not weight-sorted: {hc}"
    print(f"conceptnet mapping + weight-sort OK -> {sorted(pairs)} | has_context order={hc}")


def test_conceptnet_bad_response_no_crash():
    class _Bad:
        status_code = 500
        def json(self):
            return {}
    cn.requests.get = lambda *a, **k: _Bad()
    r = cn.conceptnet_lookup("whatever")
    assert r["found"] is False and r["edges"] == []
    print("conceptnet no-crash on HTTP 500 OK")


def test_enrich_vets_cn_edges():
    """enrich folds only the AI-kept ConceptNet edges into the node (offline fallback path)."""
    import enrich as enrich_mod
    # Offline: no key -> enrich would raise SystemError_. Instead exercise _draft_for directly,
    # simulating an AI result that keeps one edge and drops another.
    unit = {
        "term": "spring", "sentence": "Flowers bloom in the spring.", "source": "demo",
        "senses": [{"sense_id": "spring.n.01", "category": "noun.time", "pos": "noun",
                    "definition": "the season", "edges": []}],
        "cn_edges": [
            {"type": "used_for", "target": "jump", "source": "conceptnet"},        # belongs to coil sense
            {"type": "has_context", "target": "calendar", "source": "conceptnet"},  # fits season
        ],
    }
    ai = {"sense_id": "spring.n.01", "confidence": 0.9, "keep_edges": ["calendar"],
          "tags": ["Season", "season", "NATURE"]}   # mixed case + dup -> normalised, capped
    draft = enrich_mod._draft_for(unit, ai, "demo")
    targets = {(e["type"], e["target"]) for e in draft["node"]["edges"]}
    assert ("has_context", "calendar") in targets       # AI kept it
    assert ("used_for", "jump") not in targets          # AI vetoed the wrong-sense edge
    assert draft["needs_review"] is True                 # conceptnet edges force review
    assert "conceptnet_edges" in draft["ai_fields"]
    # tags: lowercased + de-duped + ordered, flagged source='ai'
    assert draft["node"]["tags"] == ["season", "nature"], draft["node"]["tags"]
    assert draft["node"]["source_map"].get("tags") == "ai"
    print(f"enrich vetting + tags OK -> kept {sorted(targets)} tags={draft['node']['tags']}")


if __name__ == "__main__":
    test_conceptnet_mapping_and_filters()
    test_conceptnet_bad_response_no_crash()
    test_enrich_vets_cn_edges()
    print("OK")
