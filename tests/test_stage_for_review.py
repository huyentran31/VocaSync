"""
Tool #10 stage_for_review — the agent's only WRITE tool writes the REVIEW QUEUE, never the graph.

Proves, offline (mocked AI):
  • stage_for_review(terms=["fed up"]) appends to the review queue (pending_drafts.json),
  • and does NOT touch data/personal_graph.json (HITL: the graph is only written at Commit).
  • the registry + MCP server now expose 10 tools, including stage_for_review.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "agent"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import enrich as en
import review_io
from _common import GRAPH_PATH


def _stash(path):
    """Move a real file aside so the test never clobbers the user's data; return the backup."""
    if os.path.exists(path):
        bak = path + ".testbak"
        shutil.move(path, bak)
        return bak
    return None


def _restore(path, bak):
    for extra in (path, path + ".bak"):
        if os.path.exists(extra) and extra != bak:
            os.remove(extra)
    if bak and os.path.exists(bak):
        shutil.move(bak, path)


def test_stage_writes_review_not_graph():
    import stage_for_review as sfr

    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([
        {"term": "fed up", "sense_id": "", "collocations": ["fed up with"],
         "mnemonic": "annoyed", "pattern": "fed up with <x>", "confidence": 0.8}])

    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    try:
        # grounded: the source line actually contains the word (S16 T1)
        out = sfr.stage_for_review(terms=["fed up"], source="Charade",
                                   sentences={"fed up": "I'm fed up with waiting."})
        assert "fed up" in out["staged"], out

        # review queue (pending) got the word; editable rows expose it as 'term'
        pending = review_io.load_pending(review_io.PENDING_PATH)
        assert any(k != "_meta" for k in pending), pending
        rows = review_io.pending_to_rows(pending)
        assert "fed up" in [r["term"] for r in rows], rows

        # the personal graph was NOT written (HITL gate: only Commit writes it)
        assert not os.path.exists(GRAPH_PATH), "stage_for_review must NOT write personal_graph.json"
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH):
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(path + ".bak"):
                os.remove(path + ".bak")
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
    print("stage_for_review OK -> wrote review queue, graph untouched")


def test_stage_flags_ungrounded():
    """S17 (owner decision, supersedes S16's hard-drop): a term with no source sentence (or a
    sentence that does not contain it) is STILL STAGED but FLAGGED '⚠ ungrounded' — only the
    learner may reject a word. It also comes back in `ungrounded` with a reason so the agent
    reports it; the commit backstop (validate_edits) still blocks it until the learner supplies
    a real sentence in the (now editable) sentence column."""
    import stage_for_review as sfr

    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps([
        {"term": "fed up", "sense_id": "", "confidence": 0.8},
        {"term": "flyback", "sense_id": "", "confidence": 0.8},
    ])

    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    try:
        out = sfr.stage_for_review(
            terms=["fed up", "flyback"], source="Charade",
            sentences={"fed up": "I'm fed up with waiting."})   # no line for 'flyback'
        assert "fed up" in out["staged"], out
        assert "flyback" in out["staged"], out                   # staged too — human decides
        ung_terms = [u["term"] for u in out.get("ungrounded", [])]
        assert "flyback" in ung_terms and "fed up" not in ung_terms, out
        assert all(u.get("reason") for u in out["ungrounded"]), out

        # both reach the review queue; the ungrounded one is FLAGGED in the ⚠ ai flag cell
        rows = review_io.pending_to_rows(review_io.load_pending(review_io.PENDING_PATH))
        by_term = {r["term"]: r for r in rows}
        assert "fed up" in by_term and "flyback" in by_term, rows
        assert "ungrounded" in by_term["flyback"]["needs_review"], by_term["flyback"]
        assert "ungrounded" not in by_term["fed up"]["needs_review"], by_term["fed up"]
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH):
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(path + ".bak"):
                os.remove(path + ".bak")
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
    print("stage_for_review OK -> ungrounded 'flyback' staged FLAGGED with reason")


def test_registry_and_mcp_have_ten_tools():
    from registry import TOOLS
    assert "stage_for_review" in TOOLS, list(TOOLS)
    assert len(TOOLS) == 10, f"expected 10 tools, got {len(TOOLS)}: {list(TOOLS)}"

    import mcp_server
    listed = mcp_server._tools_list()
    names = {t["name"] for t in listed}
    assert len(listed) == 10 and "stage_for_review" in names, names
    print("registry + MCP expose 10 tools OK")


if __name__ == "__main__":
    test_stage_writes_review_not_graph()
    test_stage_flags_ungrounded()
    test_registry_and_mcp_have_ten_tools()
    print("OK")
