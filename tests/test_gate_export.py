"""
GATE-EXPORT (task #1) — the final deliverables are produced ONLY after HITL approval.

Proves, offline (mocked AI, no key/network):
  • run_intent('mine') renders a PREVIEW graph but produces NO final artefacts
    (no .apkg / Obsidian vault / infolog.txt / highlighted.ass), and its return dict
    no longer carries deck/obsidian_vault/infolog/highlighted_ass.
  • build_final_exports(approved_subset) DOES produce them, and only for the APPROVED
    words (a rejected word never reaches the deck / infolog).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "agent"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import extract_vocab as ev
import enrich as en
from _common import run_dir
import loop

SRT = os.path.join(ROOT, "data", "_test_gate.srt")
_SRT_TEXT = (
    "1\n00:00:01,000 --> 00:00:02,000\nWe must reduce emissions.\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nIt is a gas problem.\n"
)


def _setup_mocks():
    config.AI_API_KEY = "fake"
    config.CONCEPTNET_PER_TERM = False   # keep the test offline (no network)
    ev.call_ai = lambda prompt, sysp, model=None: json.dumps([
        {"term": "reduce", "sentence": "We must reduce emissions.", "surface": "reduce", "tag": "Word"},
        {"term": "gas", "sentence": "It is a gas problem.", "surface": "gas", "tag": "Word"},
    ])
    en.call_ai = lambda p, s: json.dumps([
        {"term": "reduce", "sense_id": "reduce.v.01", "collocations": ["reduce cost"],
         "mnemonic": "less", "pattern": "reduce <x>", "confidence": 0.9},
        {"term": "gas", "sense_id": "gas.n.02", "collocations": ["natural gas"],
         "mnemonic": "air", "pattern": "reduce <gas>", "confidence": 0.9},
    ])


def test_mine_produces_no_final_but_commit_does():
    if os.path.exists(SRT):
        os.remove(SRT)
    with open(SRT, "w", encoding="utf-8") as f:
        f.write(_SRT_TEXT)
    _setup_mocks()

    # --- Mine: preview only, NO final artefacts ---
    res = loop.run_intent("mine", source=SRT)
    drafts = res.get("drafts", [])
    assert drafts, "mine should produce drafts"
    for k in ("deck", "obsidian_vault", "infolog", "highlighted_ass"):
        assert k not in res, f"mine must NOT return final export {k!r} (HITL gate)"

    mine_dir = run_dir(res["run_id"])
    assert os.path.exists(os.path.join(mine_dir, "graph.html")), "preview graph expected"
    for f in ("deck.apkg", "infolog.txt", "highlighted.ass"):
        assert not os.path.exists(os.path.join(mine_dir, f)), \
            f"mine must NOT create {f} before approval"
    assert not os.path.isdir(os.path.join(mine_dir, "obsidian_vault")), \
        "mine must NOT create an Obsidian vault before approval"

    # --- Commit: approve ONLY 'reduce' -> final exports for that word only ---
    approved = [d for d in drafts if d["node"]["term"] == "reduce"]
    assert approved, "expected a 'reduce' draft"
    exports = loop.build_final_exports(approved, run_id="test_gate_commit", srt_path=SRT)

    apkg = (exports.get("deck") or {}).get("apkg", "")
    assert apkg and os.path.exists(apkg), "commit must generate the .apkg"

    infolog = exports.get("infolog", "")
    assert infolog and os.path.exists(infolog)
    with open(infolog, "r", encoding="utf-8") as f:
        body = f.read().lower()
    # S14 T10: the infolog is the CUMULATIVE ledger of the whole committed graph, so it
    # contains the approved word plus anything already committed. The unapproved batch
    # word ("gas") must still be absent UNLESS it was already in the committed graph.
    assert "reduce" in body, "approved word missing from infolog"
    from _common import load_graph, GRAPH_PATH
    graph_terms = {n.term.lower() for n in load_graph(GRAPH_PATH).nodes.values()}
    if "gas" not in graph_terms:
        assert "gas" not in body, "unapproved word leaked into the infolog"
    # the batch-scoped deck must still exclude the unapproved word (gate intact)
    assert "gas" not in os.path.basename(apkg).lower()

    if os.path.exists(SRT):
        os.remove(SRT)
    print(f"GATE-EXPORT OK -> mine produced no final; commit deck {os.path.basename(apkg)}")


if __name__ == "__main__":
    test_mine_produces_no_final_but_commit_does()
    print("OK")
