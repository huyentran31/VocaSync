"""enrich: deterministic-first behavior with a mocked AI call."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import enrich as en
from wordnet_lookup import wordnet_lookup
from schema import Node


def _units(term, sentence):
    return [{"term": term, "sentence": sentence, "senses": wordnet_lookup(term)["senses"], "source": "demo"}]


def test_valid_sense_pick_grounds_edges_and_flags_ai():
    config.AI_API_KEY = "fake"
    en._POLYSEMY_REVIEW_MIN = 99   # isolate: this case tests sense-pick, not the polysemy flag
    # gas.n.01 is the gaseous-state sense
    en.call_ai = lambda p, s: json.dumps([{
        "term": "gas", "sense_id": "gas.n.02",
        "collocations": ["natural gas", "gas emissions"],
        "mnemonic": "gas = ghostly air", "pattern": "reduce <gas>", "confidence": 0.9,
    }])
    d = en.enrich(_units("gas", "reduce carbon gas emissions"), source="demo")[0]
    node = Node(**d["node"])                       # must validate against schema
    assert node.sense_id == "gas.n.02"
    assert node.source_map["sense_id"] == "ai"     # the CHOICE is AI's
    assert node.source_map.get("collocations") == "ai"
    assert all(e.source == "wordnet" for e in node.edges)   # edges grounded
    assert node.source_map.get("edges") == "wordnet"
    assert node.key == "gas#gas.n.02"
    assert d["needs_review"] is False
    print("valid-sense:", node.key, "| edges", len(node.edges), "| ai_fields", d["ai_fields"])


def test_out_of_list_sense_falls_back_and_flags_review():
    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([{
        "term": "gas", "sense_id": "gas.n.99",  # not a real candidate
        "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.95,
    }])
    d = en.enrich(_units("gas", "step on the gas"), source="demo")[0]
    node = Node(**d["node"])
    assert node.sense_id == "gas.n.01"   # fell back to most common sense[0]
    assert d["needs_review"] is True     # forced review
    assert d["confidence"] <= 0.5
    print("fallback:", node.sense_id, "needs_review", d["needs_review"])


def test_low_confidence_forces_review():
    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([{
        "term": "reduce", "sense_id": "reduce.v.01",
        "collocations": ["reduce cost"], "mnemonic": "m", "pattern": "p", "confidence": 0.4,
    }])
    d = en.enrich(_units("reduce", "we reduce cost"), source="demo")[0]
    assert d["needs_review"] is True
    print("low-conf review:", d["confidence"])


def test_oov_accepts_ai_definition():
    """S15 T2: an OOV term (no WordNet senses) gets the AI definition, flagged source=ai."""
    config.AI_API_KEY = "fake"
    en._POLYSEMY_REVIEW_MIN = 99
    en.call_ai = lambda p, s: json.dumps([{
        "term": "get a divorce", "sense_id": None,
        "definition": "to legally end a marriage",
        "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.95,
    }])
    d = en.enrich([{"term": "get a divorce", "sentence": "We are getting a divorce.",
                    "senses": [], "source": "demo"}], source="demo")[0]
    node = Node(**d["node"])
    assert node.definition == "to legally end a marriage"
    assert node.source_map.get("definition") == "ai"
    assert "definition" in d["ai_fields"]
    assert d["needs_review"] is True            # OOV always review
    assert d["confidence"] <= 0.3               # OOV confidence capped
    # no AI definition -> definition stays None (old behaviour)
    en.call_ai = lambda p, s: json.dumps([{
        "term": "make it work", "sense_id": None,
        "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.9,
    }])
    d2 = en.enrich([{"term": "make it work", "sentence": "I tried to make it work.",
                     "senses": [], "source": "demo"}], source="demo")[0]
    assert Node(**d2["node"]).definition is None
    print("OOV AI definition OK")


def test_low_confidence_threshold_env_knob():
    """S14 T6: LOW_CONFIDENCE_THRESHOLD is read from the environment for real."""
    import importlib
    config.AI_API_KEY = "fake"
    os.environ["LOW_CONFIDENCE_THRESHOLD"] = "0.9"
    try:
        importlib.reload(en)
        assert en._LOW_CONF == 0.9
        en._POLYSEMY_REVIEW_MIN = 99
        en.call_ai = lambda p, s: json.dumps([{
            "term": "reduce", "sense_id": "reduce.v.01",
            "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.8,
        }])
        d = en.enrich(_units("reduce", "we reduce cost"), source="demo")[0]
        assert d["needs_review"] is True     # 0.8 < 0.9 threshold from env
    finally:
        del os.environ["LOW_CONFIDENCE_THRESHOLD"]
        importlib.reload(en)                 # restore default 0.7 for later tests
    print("env threshold knob OK")


def test_polysemy_forces_review():
    """S14 T7: >= POLYSEMY_REVIEW_MIN candidate senses -> needs_review, confidence untouched."""
    config.AI_API_KEY = "fake"
    en._POLYSEMY_REVIEW_MIN = 5
    # "run" is heavily polysemous (>> 5 senses) — high confidence must NOT exempt it
    senses = wordnet_lookup("run")["senses"]
    assert len(senses) >= 5
    en.call_ai = lambda p, s: json.dumps([{
        "term": "run", "sense_id": senses[0]["sense_id"],
        "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.95,
    }])
    d = en.enrich([{"term": "run", "sentence": "I run every day", "senses": senses,
                    "source": "demo"}], source="demo")[0]
    assert d["needs_review"] is True
    assert d["confidence"] == 0.95           # flag only, no confidence change
    assert "polysemy" in d["ai_fields"]
    # a 2-sense term with high confidence keeps needs_review False
    two = wordnet_lookup("gas")["senses"][:2]
    en.call_ai = lambda p, s: json.dumps([{
        "term": "gas", "sense_id": two[0]["sense_id"],
        "collocations": [], "mnemonic": "", "pattern": "", "confidence": 0.9,
    }])
    d2 = en.enrich([{"term": "gas", "sentence": "reduce gas", "senses": two,
                     "source": "demo"}], source="demo")[0]
    assert d2["needs_review"] is False
    print("polysemy flag OK")


if __name__ == "__main__":
    test_valid_sense_pick_grounds_edges_and_flags_ai()
    test_out_of_list_sense_falls_back_and_flags_review()
    test_low_confidence_forces_review()
    test_oov_accepts_ai_definition()
    test_low_confidence_threshold_env_knob()
    test_polysemy_forces_review()
    print("OK")
