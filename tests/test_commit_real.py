"""
REAL commit path (S12 non-excel) — calls app.commit_approved directly, no mirror.

test_hitl / test_validate_partition each mirror the commit loop; a mirror can drift from
the real function. This test exercises the ACTUAL app.commit_approved(review_df) — the row
edit-application + validate_edits(T4) + partition + single graph write — with the module's
side-effecting tail sandboxed so it never touches the real personal_graph.json:

  * app.GRAPH_PATH        -> a temp file (load_graph starts empty; save_graph writes here)
  * review_io.load_pending-> returns our staged pending dict (avoids the real data file)
  * app.call_tool         -> stub (skip real graph render)
  * app.build_final_exports-> stub (skip Anki/vault build)
  * app.st                -> shim with a plain-dict session_state

Proves, offline/no-key: a valid approved row commits + writes the graph; a fabricated
sense_id is PARTITIONED (held back with a reason) and, when it's the only row, the graph
is NOT written. Uses review_io.pending_to_rows -> DataFrame, exactly what st.data_editor feeds.
"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import pandas as pd
import config
import enrich as en
from wordnet_lookup import wordnet_lookup
import review_io
import app  # importing runs the UI once in bare mode (read-only) — same as test_app_boot

TMP_GRAPH = os.path.join(ROOT, "data", "_test_commit_real_graph.json")


def _pending():
    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([{
        "term": "gas", "sense_id": "gas.n.02",
        "collocations": ["natural gas"], "mnemonic": "gas=air",
        "pattern": "reduce <gas>", "confidence": 0.9}])
    senses = wordnet_lookup("gas")["senses"]
    drafts = en.enrich([{"term": "gas", "sentence": "reduce carbon gas emissions",
                         "senses": senses, "source": "demo"}])
    tmp_pend = os.path.join(ROOT, "data", "_test_commit_real_pending.json")
    review_io.export_review(drafts, pending_path=tmp_pend)
    pending = review_io.load_pending(tmp_pend)
    os.remove(tmp_pend)
    return pending


def _sandbox(pending):
    """Point the real commit at temp/stubbed side effects; return a restore() callable."""
    saved = {"GRAPH_PATH": app.GRAPH_PATH, "load_pending": review_io.load_pending,
             "call_tool": app.call_tool, "build_final_exports": app.build_final_exports,
             "st": app.st}
    app.GRAPH_PATH = TMP_GRAPH
    review_io.load_pending = lambda *a, **k: pending
    app.call_tool = lambda *a, **k: "graph.html"
    app.build_final_exports = lambda *a, **k: {}
    app.st = types.SimpleNamespace(session_state={})

    def restore():
        app.GRAPH_PATH = saved["GRAPH_PATH"]
        review_io.load_pending = saved["load_pending"]
        app.call_tool = saved["call_tool"]
        app.build_final_exports = saved["build_final_exports"]
        app.st = saved["st"]
    return restore


def _df(pending):
    return pd.DataFrame(review_io.pending_to_rows(pending), columns=review_io.COLUMNS)


def test_real_commit_valid_and_partition():
    if os.path.exists(TMP_GRAPH):
        os.remove(TMP_GRAPH)
    pending = _pending()
    restore = _sandbox(pending)
    try:
        # --- valid approve -> real commit writes the (temp) graph ---
        df = _df(pending)
        df.loc[0, "status"] = "approved"
        res = app.commit_approved(df)
        assert res["committed"] and not res["invalid"], res
        assert os.path.exists(TMP_GRAPH), "valid row must write the graph (single write point)"
        saved_graph = json.load(open(TMP_GRAPH, encoding="utf-8"))
        assert any("gas" in k for k in saved_graph.get("nodes", {})), saved_graph.get("nodes")
        os.remove(TMP_GRAPH)

        # --- fabricated sense_id -> PARTITION: held back, graph NOT written ---
        df = _df(pending)
        df.loc[0, "status"] = "approved"
        df.loc[0, "sense_id"] = "totallyfake.n.99"
        res = app.commit_approved(df)
        assert not res["committed"], "fabricated sense_id must not commit"
        assert res["invalid"] and "sense_id" in res["invalid"][0]["reason"], res
        assert not os.path.exists(TMP_GRAPH), "no valid row -> graph must NOT be written"
    finally:
        restore()
        for p in (TMP_GRAPH,):
            if os.path.exists(p):
                os.remove(p)
    print("real commit_approved OK -> valid commits + writes graph; fabricated sense_id held back")


def test_real_commit_term_edit_rekeys_and_drops_old():
    """S19 OPEN-5: editing the term (fixing a distorted headword like 'be all over the place'
    -> 'all over the place') rekeys the node to term#sense_id AND drops the stale old-key node,
    so the graph keeps no orphan / the deck no duplicate card. Two commits: the first plants the
    old-key node, the second renames it and must remove the original."""
    if os.path.exists(TMP_GRAPH):
        os.remove(TMP_GRAPH)
    pending = _pending()
    old_key = next(k for k in pending if k != "_meta")     # 'gas#gas.n.02'
    restore = _sandbox(pending)
    try:
        # 1) commit as-is so the OLD-key node exists in the graph
        df = _df(pending)
        df.loc[0, "status"] = "approved"
        app.commit_approved(df)
        g = json.load(open(TMP_GRAPH, encoding="utf-8"))
        assert old_key in g.get("nodes", {}), ("setup: old key must commit", list(g["nodes"]))

        # 2) fix the headword -> rekey + drop the old node
        df = _df(pending)
        df.loc[0, "status"] = "approved"
        df.loc[0, "term"] = "natural gas"
        res = app.commit_approved(df)
        assert res["committed"], res
        g = json.load(open(TMP_GRAPH, encoding="utf-8"))
        nodes = g.get("nodes", {})
        new_key = f"natural gas#{old_key.split('#', 1)[1]}"
        assert new_key in nodes, ("renamed node must be committed", new_key, list(nodes))
        assert old_key not in nodes, ("stale old-key node must be dropped", old_key, list(nodes))
    finally:
        restore()
        if os.path.exists(TMP_GRAPH):
            os.remove(TMP_GRAPH)
    print("OPEN-5 OK -> term edit rekeys node + drops the stale old-key node")


if __name__ == "__main__":
    test_real_commit_valid_and_partition()
    test_real_commit_term_edit_rekeys_and_drops_old()
    print("OK")
