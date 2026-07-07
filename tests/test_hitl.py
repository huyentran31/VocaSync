"""
HITL round-trip (GĐ4 task 3; NON-EXCEL flow) — the in-app review df is the checkpoint.

Proves, with NO network/key and WITHOUT importing streamlit:
  enrich drafts -> export_review (pending_drafts.json, NO review.xlsx)
  -> pending_to_rows() builds the editable rows the app shows in st.data_editor
  -> human marks ONE row status=approved (we edit the row dict, like editing the data_editor)
  -> commit core: rows -> approved rows -> upsert -> save_graph -> build_render_graph
And the HITL GATE: a draft is NOT in the graph until it is approved + committed.
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
from build_render_graph import build_render_graph
from schema import PersonalGraph, Node
import review_io

GRAPH = os.path.join(ROOT, "data", "_test_hitl_graph.json")
PEND = os.path.join(ROOT, "data", "_test_pending.json")


def test_reconcile_surfaces_agent_staged_words():
    """S17 regression: the review table must MIRROR the pending queue, not a stale snapshot.

    Bug it guards: review_df was cached in session_state and only rebuilt when None, so words
    the agent staged mid-conversation (appended to pending on disk) never appeared — the table
    stuck at the last Mine's rows while the agent truthfully reported 'staged'. reconcile_rows
    adds new disk rows and KEEPS the learner's in-progress edits on existing rows."""
    import pandas as pd
    # a cached session df with ONE row the learner already edited (status set)
    cached = pd.DataFrame([{
        "#": 1, "status": "rejected", "term": "gas", "sentence": "reduce carbon gas",
        "definition": "a fuel", "word_type": "word", "sense_id": "gas.n.02", "tags": "",
        "needs_review": "", "ai_fields": "", "confidence": "0.90", "comment": "", "key": "gas#gas.n.02"}],
        columns=review_io.COLUMNS)
    # disk now has the old word PLUS two the agent just staged
    pending = {
        "gas#gas.n.02": {"node": {"key": "gas#gas.n.02", "term": "gas", "definition": "a fuel",
                                  "sense_id": "gas.n.02", "occurrences": [{"sentence": "reduce carbon gas"}]},
                         "confidence": 0.9, "needs_review": False, "ai_fields": []},
        "go on#go_on.v.01": {"node": {"key": "go on#go_on.v.01", "term": "go on", "definition": "continue",
                                      "occurrences": [{"sentence": "go on, tell me"}]},
                             "confidence": 0.5, "needs_review": True, "ai_fields": ["ungrounded"]},
        "give up#give_up.v.01": {"node": {"key": "give up#give_up.v.01", "term": "give up", "definition": "quit",
                                          "occurrences": [{"sentence": "don't give up"}]},
                                 "confidence": 0.9, "needs_review": False, "ai_fields": []},
    }
    out = review_io.reconcile_rows(cached, pending)
    by_term = {r["term"]: r for r in out.to_dict("records")}
    assert set(by_term) == {"gas", "go on", "give up"}, by_term          # agent words now visible
    assert by_term["gas"]["status"] == "rejected", "learner edit must survive"   # edit kept
    assert "ungrounded" in by_term["go on"]["needs_review"], by_term["go on"]   # flag from disk
    assert list(out["#"]) == [1, 2, 3], "# renumbered continuously"
    print("reconcile OK -> agent-staged words surface, learner edits preserved")


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
    """Mirror app.commit_approved (without streamlit) — the single graph write."""
    pending = review_io.load_pending(pending_path)
    graph = PersonalGraph.load(graph_path)
    committed = []
    for r in rows:
        if str(r.get("status", "")).strip().lower() != "approved":
            continue
        pend = pending.get(str(r.get("key", "")).strip())
        if not pend:
            continue
        graph.upsert(Node(**pend["node"]))
        committed.append(r.get("key"))
    if committed:
        graph.save(graph_path)
    return graph, committed


def test_hitl_gate_and_commit():
    for p in (GRAPH, PEND):
        if os.path.exists(p):
            os.remove(p)

    drafts = _drafts()
    out = review_io.export_review(drafts, pending_path=PEND)
    assert out == PEND and os.path.exists(PEND)
    # NON-EXCEL: export_review must NOT create a review.xlsx anywhere
    assert not os.path.exists(os.path.join(ROOT, "data", "review.xlsx"))

    # editable rows (what the app feeds st.data_editor) carry the HANDOVER §5.3 columns
    rows = review_io.pending_to_rows(review_io.load_pending(PEND))
    assert rows, "pending_to_rows should yield at least one row"
    for col in ("term", "word_type", "sense_id", "definition", "confidence",
                "needs_review", "tags", "status", "comment", "key"):
        assert col in rows[0], f"missing column {col}"

    # HITL GATE: nothing approved yet -> commit writes NOTHING to the graph
    g0, committed0 = _commit_core(GRAPH, rows, PEND)
    assert committed0 == [] and not os.path.exists(GRAPH)

    # human approves the single row (edit the row like a teacher would in the data_editor)
    rows[0]["status"] = "approved"

    # commit -> node now in graph + graph.html rendered
    g1, committed1 = _commit_core(GRAPH, rows, PEND)
    assert committed1, "approved row should have committed"
    assert g1.recall("gas")["as_main_node"] is not None
    html = build_render_graph(g1, run_id="test_hitl")
    assert os.path.exists(html)

    for p in (GRAPH, PEND):
        if os.path.exists(p):
            os.remove(p)
    print(f"HITL OK -> committed {committed1}, graph {os.path.basename(html)}")


def test_append_dedup_and_meta():
    """export_review append dedups by key; _meta merges (never overwritten with empty)."""
    for p in (PEND,):
        if os.path.exists(p):
            os.remove(p)
    drafts = _drafts()

    review_io.export_review(drafts, pending_path=PEND, srt_path="demo.srt", source="demo")
    p1 = review_io.load_pending(PEND)
    keys1 = [k for k in p1 if k != "_meta"]
    assert keys1 and p1["_meta"]["srt_path"] == "demo.srt"

    # append the SAME drafts -> no duplicate keys; _meta srt_path preserved (empty not clobbered)
    review_io.export_review(drafts, pending_path=PEND, mode="append")
    p2 = review_io.load_pending(PEND)
    keys2 = [k for k in p2 if k != "_meta"]
    assert sorted(keys2) == sorted(keys1), "append must dedup by key"
    assert p2["_meta"]["srt_path"] == "demo.srt", "empty srt_path must not clobber prior _meta"

    if os.path.exists(PEND):
        os.remove(PEND)
    print("append dedup + _meta OK ->", keys2, p2["_meta"])


if __name__ == "__main__":
    test_hitl_gate_and_commit()
    test_append_dedup_and_meta()
    test_reconcile_surfaces_agent_staged_words()
    print("OK")
