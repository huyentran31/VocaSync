"""
Smoke tests for the Day-4 evals (offline, no key / network):
  • sense_eval runs over the golden set and reports an accuracy + writes a result file.
  • mine_corrections clusters a synthetic review.xlsx of rejected/needs_revision rows.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy"),
          os.path.join(ROOT, "evals")):
    sys.path.insert(0, p)

import config
import pandas as pd
import review_io
import sense_eval
import mine_corrections


def test_sense_eval_offline():
    config.AI_API_KEY = ""        # force offline (deterministic) mode
    out = sense_eval.run_eval()
    s = out["summary"]
    assert s["mode"] == "offline"
    assert s["n"] >= 10
    assert 0.0 <= s["exact_match_accuracy"] <= 1.0
    assert os.path.exists(out["out_path"])
    print("sense_eval:", s["exact_match_accuracy"])


def test_mine_corrections_synthetic():
    xlsx = os.path.join(ROOT, "data", "_test_corrections.xlsx")
    rows = [
        {"term": "gas", "sense_id": "gas.n.02", "definition": "fuel for cars",
         "confidence": "0.4", "needs_review": "TRUE", "ai_fields": "sense_id",
         "status": "rejected", "comment": "wrong sense, this is physics not gasoline", "key": "gas#x"},
        {"term": "spring", "sense_id": "spring.n.02", "definition": "metal coil",
         "confidence": "0.5", "needs_review": "TRUE", "ai_fields": "sense_id",
         "status": "rejected", "comment": "wrong sense, means the season here", "key": "spring#x"},
        {"term": "deadline", "sense_id": "", "definition": "",
         "confidence": "0.3", "needs_review": "TRUE", "ai_fields": "",
         "status": "needs_revision", "comment": "no wordnet entry, missing definition", "key": "deadline#x"},
        {"term": "API", "sense_id": "", "definition": "",
         "confidence": "0.2", "needs_review": "TRUE", "ai_fields": "",
         "status": "needs_revision", "comment": "no wordnet entry, missing definition", "key": "api#x"},
    ]
    pd.DataFrame(rows, columns=review_io.COLUMNS).to_excel(
        xlsx, index=False, sheet_name="Review", engine="openpyxl")

    out = mine_corrections.mine(xlsx, k=2)
    assert out["n"] == 4
    assert len(out["clusters"]) >= 1
    assert sum(c["size"] for c in out["clusters"]) == 4
    os.remove(xlsx)
    print("mine_corrections clusters:", [c["failure_mode_hint"] for c in out["clusters"]])


if __name__ == "__main__":
    test_sense_eval_offline()
    test_mine_corrections_synthetic()
    print("OK")
