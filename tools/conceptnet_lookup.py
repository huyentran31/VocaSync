"""
conceptnet_lookup.py — Tool #9 (read, DETERMINISTIC, online).

WordNet is the ontology backbone (is_a / part_of / synonym) but it is SPARSE for
the "life-context" layer and for modern/OOV terms (e.g. `touchpad` part_of `laptop`,
`key` used_for `open door`). ConceptNet fills exactly that gap.

Like wordnet_lookup, this returns grounded `Edge`s with source="conceptnet" so the
LLM cannot invent them — `enrich` later VETS which of these fit the chosen sense.

Design rules (locked in spec):
  • language : English only (drop edges that touch a non-`en` node)
  • direction: keep edges whose START is the query term (tusk PartOf elephant, NOT reverse)
  • weight   : drop edges below config.CONCEPTNET_MIN_WEIGHT (precision filter)
  • relations: PartOf→part_of · UsedFor→used_for · HasContext→has_context ·
               Synonym→synonym · Antonym→antonym · IsA→is_a
  • DROP /r/RelatedTo entirely — it is the noisiest ConceptNet relation.

On any network/parse error: return found=False with edges=[] (no-crash, AGENTS.md §4).
"""

from __future__ import annotations

import requests

from _common import log_tool_call   # import FIRST: puts repo root + legacy/ on sys.path
import config                       # (config shim lives in legacy/)
from schema import Edge, EDGE_TYPES

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "term": {"type": "string", "description": "Word/phrase to look up in ConceptNet (English)."},
        "min_weight": {"type": "number", "description": "Drop edges below this weight (default config.CONCEPTNET_MIN_WEIGHT)."},
        "max_per_relation": {"type": "integer", "description": "Cap edges kept per relation (default 6)."},
    },
    "required": ["term"],
}

# ConceptNet relation id -> our locked Edge.type. /r/RelatedTo is deliberately absent.
_REL_MAP = {
    "/r/PartOf": "part_of",
    "/r/UsedFor": "used_for",
    "/r/HasContext": "has_context",
    "/r/Synonym": "synonym",
    "/r/Antonym": "antonym",
    "/r/IsA": "is_a",
}


def _term_id(term: str) -> str:
    """ConceptNet node id form: 'open door' -> 'open_door'."""
    return term.strip().lower().replace(" ", "_")


def _node_term(node_id: str) -> str:
    """'/c/en/tusk' or '/c/en/tusk/n' -> 'tusk' (the lemma part, index 3)."""
    parts = (node_id or "").split("/")
    return parts[3] if len(parts) > 3 else ""


def conceptnet_lookup(term: str, min_weight: float | None = None,
                      max_per_relation: int = 6) -> dict:
    """Return {"term","found","edges":[Edge-as-dict ...]} ready to merge into a Node.

    Edges are direction-filtered (START == term), language-filtered (en), weight-filtered,
    and relation-mapped. Never raises — a dead API just yields found=False.
    """
    min_weight = config.CONCEPTNET_MIN_WEIGHT if min_weight is None else float(min_weight)
    args = {"term": term, "min_weight": min_weight, "max_per_relation": max_per_relation}

    tid = _term_id(term)
    url = f"{config.CONCEPTNET_ENDPOINT}/c/en/{tid}"
    try:
        resp = requests.get(url, params={"limit": config.CONCEPTNET_MAX_EDGES},
                            timeout=config.CONCEPTNET_TIMEOUT)
        if resp.status_code != 200:
            log_tool_call("conceptnet_lookup", args, error=f"HTTP {resp.status_code}")
            return {"term": term, "found": False, "edges": []}
        data = resp.json()
    except Exception as e:  # network/timeout/JSON — stay non-fatal
        log_tool_call("conceptnet_lookup", args, error=f"conceptnet unavailable: {e}")
        return {"term": term, "found": False, "edges": []}

    # 1) collect valid candidates with their weights
    cands = []                              # (weight, etype, target)
    for e in data.get("edges", []) or []:
        rel = (e.get("rel") or {}).get("@id")
        etype = _REL_MAP.get(rel)
        if not etype:                       # unmapped rel (incl. /r/RelatedTo) -> skip
            continue

        start, end = e.get("start") or {}, e.get("end") or {}
        # English only — both ends must be /c/en/*
        if start.get("language") != "en" or end.get("language") != "en":
            continue
        # Direction: the query term must be the START (so target = the WHOLE/purpose/context)
        if _node_term(start.get("@id", "")) != tid:
            continue

        weight = float(e.get("weight", 1.0) or 0.0)
        if weight < min_weight:
            continue

        target = (end.get("label") or _node_term(end.get("@id", ""))).strip()
        if not target or target.lower() == term.strip().lower():
            continue
        cands.append((weight, etype, target))

    # 2) STRONGEST FIRST: sort by weight desc so the top has_context/used_for wins as the
    #    graph group label, then dedup + cap per relation (order now reflects confidence).
    cands.sort(key=lambda c: c[0], reverse=True)
    edges, seen, per_rel = [], set(), {}
    for weight, etype, target in cands:
        key = (etype, target.lower())
        if key in seen:
            continue
        if per_rel.get(etype, 0) >= max_per_relation:
            continue
        seen.add(key)
        per_rel[etype] = per_rel.get(etype, 0) + 1
        edges.append(Edge(type=etype, target=target, source="conceptnet"))

    assert all(e.type in EDGE_TYPES for e in edges)  # locked vocabulary
    out = {"term": term, "found": bool(edges),
           "edges": [e.model_dump() for e in edges]}
    log_tool_call("conceptnet_lookup", args, result={"edges": len(edges)})
    return out


if __name__ == "__main__":
    import json
    import sys

    t = sys.argv[1] if len(sys.argv) > 1 else "tusk"
    r = conceptnet_lookup(t)
    print(f"{t}: found={r['found']} edges={len(r['edges'])}")
    print(json.dumps(r["edges"], ensure_ascii=False, indent=2))
