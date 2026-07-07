"""
UI smoke test (task #7) — the Streamlit app boots headless with no exception.

Uses Streamlit's AppTest to execute app.py end-to-end (no browser). Proves the dark theme
+ masthead + multi-turn chat scaffolding + gated downloads render without raising, even with
NO AI key (the no-key banner path).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


def test_app_boots_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    # masthead is present in the rendered markdown
    blob = " ".join(str(getattr(m, "value", "")) for m in at.markdown)
    assert "masthead" in blob or "VocabGraph-Agent" in blob
    print("app boot OK -> no exception; masthead rendered")


def test_approve_undecided_respects_human_choices():
    """S14 T5: Approve All fills only BLANK statuses; Approve confident only FALSE-blank."""
    import pandas as pd
    # app.py runs Streamlit code at import; instead pull the pure function via exec of its def
    src = open(APP, "r", encoding="utf-8").read()
    ns = {}
    start = src.index("def approve_undecided")
    end = src.index("def commit_approved")
    exec(src[start:end], ns)
    approve_undecided = ns["approve_undecided"]

    # `needs_review` cell now holds the flag REASONS (empty = no concern); legacy
    # "FALSE" from an old session must still count as no-concern.
    df = pd.DataFrame([
        {"term": "a", "needs_review": "polysemy", "status": "rejected"},
        {"term": "b", "needs_review": "definition(ai)", "status": ""},
        {"term": "c", "needs_review": "", "status": ""},
        {"term": "d", "needs_review": "FALSE", "status": ""},   # legacy value
    ])
    all_df, n_all = approve_undecided(df)
    assert n_all == 3
    assert all_df.loc[0, "status"] == "rejected"          # human decision untouched
    assert all_df.loc[1, "status"] == all_df.loc[2, "status"] == all_df.loc[3, "status"] == "approved"

    conf_df, n_conf = approve_undecided(df, confident_only=True)
    assert n_conf == 2
    assert conf_df.loc[0, "status"] == "rejected"
    assert conf_df.loc[1, "status"] == ""                 # flagged-blank stays undecided
    assert conf_df.loc[2, "status"] == "approved"         # empty flag = confident
    assert conf_df.loc[3, "status"] == "approved"         # legacy FALSE = confident
    print("approve_undecided OK -> rejected kept, flag reasons drive confident")


if __name__ == "__main__":
    test_app_boots_without_exception()
    test_approve_undecided_respects_human_choices()
    print("OK")
