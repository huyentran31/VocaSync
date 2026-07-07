"""
Double-check validate + PARTITION at commit (S12 T4) — NON-EXCEL (df-session) path.

Proves, with NO network/key and WITHOUT running the streamlit UI:
  * app.validate_edits() is a deterministic gate: it accepts a clean node and
    rejects a fabricated sense_id / bad word_type / empty definition (no AI).
  * The commit loop PARTITIONS: a row whose human edit is invalid is held back
    (with a reason) and does NOT reach the graph, while valid rows in the same
    batch still commit. Mirrors app.commit_approved's edit-apply + validate step
    (the real app.validate_edits is imported and exercised — only the commit loop
    is mirrored, same pattern as test_hitl._commit_core).
  * Review rows come from review_io.pending_to_rows (what st.data_editor shows);
    edits are made directly on the row dicts (as the learner would in the editor).

Invariant checked: personal_graph.json is written ONLY when a valid row commits.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import enrich as en
from wordnet_lookup import wordnet_lookup
from schema import PersonalGraph, Node
import review_io

# app.validate_edits is a pure deterministic function; importing app runs the UI
# module once in bare mode (read-only) — acceptable, same as test_app_boot.
from app import validate_edits

GRAPH = os.path.join(ROOT, "data", "_test_partition_graph.json")
PEND = os.path.join(ROOT, "data", "_test_partition_pending.json")


def _drafts():
    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([{
        "term": "gas", "sense_id": "gas.n.02",
        "collocations": ["natural gas"], "mnemonic": "gas=air",
        "pattern": "reduce <gas>", "confidence": 0.9}])
    senses = wordnet_lookup("gas")["senses"]
    return en.enrich([{"term": "gas", "sentence": "reduce carbon gas emissions",
                       "senses": senses, "source": "demo"}])


def _commit_core(graph_path, rows, pending_path):
    """Mirror app.commit_approved: apply human edits, run validate_edits, PARTITION.

    Returns (committed_keys, invalid_rows). The graph is saved ONLY when at least
    one valid row commits (the single write point).
    """
    pending = review_io.load_pending(pending_path)
    graph = PersonalGraph.load(graph_path)
    committed, invalid = [], []
    for r in rows:
        if str(r.get("status", "")).strip().lower() != "approved":
            continue
        key = str(r.get("key", "")).strip()
        pend = pending.get(key)
        if not pend:
            continue
        node_dict = dict(pend["node"])
        new_sense = str(r.get("sense_id", "")).strip()
        if new_sense and new_sense != (node_dict.get("sense_id") or ""):
            node_dict["sense_id"] = new_sense          # human edit (may be fabricated)
        problems = validate_edits(node_dict)           # <-- the real T4 gate
        if problems:
            invalid.append({"key": key, "term": node_dict.get("term", ""),
                            "reason": "; ".join(problems)})
            continue
        graph.upsert(Node(**node_dict))
        committed.append(key)
    if committed:
        graph.save(graph_path)
    return committed, invalid


# a grounded occurrence (non-empty source sentence) — required by the S16 T2 gate
_OCC = [{"source": "demo", "sentence": "reduce carbon gas emissions"}]


def test_validate_edits_unit():
    """The gate in isolation: clean node passes; each defect is reported."""
    assert validate_edits({"word_type": "word", "definition": "a fuel",
                           "sense_id": "gas.n.02", "term": "gas", "tags": ["fuel"],
                           "occurrences": _OCC}) == []
    assert validate_edits({"word_type": "word", "definition": "x",
                           "sense_id": "notreal.n.99", "term": "gas", "tags": [],
                           "occurrences": _OCC})
    bad = validate_edits({"word_type": "verbphrase", "definition": "",
                          "sense_id": "", "term": "gas", "tags": [], "occurrences": _OCC})
    assert any("word_type" in r for r in bad) and any("definition" in r for r in bad)
    print("validate_edits unit OK")


def test_ungrounded_node_held_back():
    """S16 T2: a node complete on definition/word_type but with NO grounded occurrence
    (occurrences=[] or all sentences blank) is held back with the grounding reason; the
    same node WITH a source sentence passes."""
    base = {"word_type": "word", "definition": "a fuel", "sense_id": "gas.n.02",
            "term": "gas", "tags": ["fuel"]}
    problems = validate_edits({**base, "occurrences": []})
    assert any("no grounded occurrence" in r for r in problems), problems
    # a blank-sentence occurrence is still ungrounded
    assert any("no grounded occurrence" in r
               for r in validate_edits({**base, "occurrences": [{"sentence": ""}]}))
    # with a real source sentence it commits (no reasons)
    assert validate_edits({**base, "occurrences": _OCC}) == []
    print("ungrounded node held back OK")


def test_partition_holds_back_invalid_edit():
    for p in (GRAPH, PEND):
        if os.path.exists(p):
            os.remove(p)

    drafts = _drafts()
    review_io.export_review(drafts, pending_path=PEND)
    rows = review_io.pending_to_rows(review_io.load_pending(PEND))
    assert rows

    # --- case 1: valid approve -> commits, graph written ---
    rows[0]["status"] = "approved"
    committed, invalid = _commit_core(GRAPH, rows, PEND)
    assert committed and not invalid, (committed, invalid)
    assert os.path.exists(GRAPH), "valid row must write the graph"
    os.remove(GRAPH)

    # --- case 2: same row approved but sense_id edited to a FABRICATED sense ---
    rows[0]["sense_id"] = "totallyfake.n.99"
    committed, invalid = _commit_core(GRAPH, rows, PEND)
    assert not committed, "fabricated sense_id must NOT commit"
    assert invalid and "sense_id" in invalid[0]["reason"], invalid
    assert not os.path.exists(GRAPH), "no valid row -> graph must NOT be written"

    for p in (GRAPH, PEND):
        if os.path.exists(p):
            os.remove(p)
    print("partition OK -> valid commits; fabricated sense_id held back:", invalid[0]["reason"])


if __name__ == "__main__":
    test_validate_edits_unit()
    test_ungrounded_node_held_back()
    test_partition_holds_back_invalid_edit()
    print("OK")
