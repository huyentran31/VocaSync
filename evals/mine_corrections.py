"""
mine_corrections.py — Eval #4 (Day-4, the ⭐ one): K-Means mining of human corrections.

HITL corrections are a free, high-signal eval source: every row the learner marks
`rejected` or `needs_revision` is a place the agent got something wrong. Clustering those
rows surfaces the TOP FAILURE MODES to fix next (SECURITY_EVAL.md / REVIEW_CHECKLIST H).

Since the live review queue is now in-app (st.data_editor, non-Excel), this eval reads a
`corrections.xlsx` the learner exports/keeps of their reviewed rows — a standalone offline
input, decoupled from the app's review flow (see DEFAULT_CORRECTIONS below).

Pipeline (deterministic, offline, NO new dependency):
  rejected/needs_revision rows -> text = term + definition + comment
  -> TF-IDF vectors (hand-rolled) -> L2 normalize -> K-Means (pure numpy, seeded)
  -> per cluster: size, representative terms, top distinguishing tokens (= the mode).

We avoid scikit-learn on purpose (anti-slopsquatting / dependency-light): a small,
seeded numpy K-Means is fully reproducible and easy to audit.

Usage:  python evals/mine_corrections.py [corrections.xlsx] [k]
Output: prints clusters + writes evals/results/corrections_<run>.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NON-EXCEL: the live review queue no longer persists as review.xlsx (it lives in the app's
# st.data_editor). This offline eval reads a corrections spreadsheet the learner exports/keeps,
# so it stays self-contained (plain pandas) and does not depend on the review_io Excel helpers.
DEFAULT_CORRECTIONS = os.path.join(_ROOT, "data", "corrections.xlsx")
RESULTS_DIR = os.path.join(_ROOT, "evals", "results")
_TARGET_STATUSES = {"rejected", "needs_revision"}
_STOP = set("the a an of to in on for and or is are was were be been with as at by it this that "
            "not no its their his her our your my you we they i he she them".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z']+", str(text).lower())
            if t not in _STOP and len(t) > 2]


def _tfidf(docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Hand-rolled TF-IDF matrix (rows=docs). Returns (matrix, vocab)."""
    vocab = sorted({t for d in docs for t in d})
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(docs)
    tf = np.zeros((n, len(vocab)))
    for r, d in enumerate(docs):
        for t in d:
            tf[r, idx[t]] += 1.0
        if d:
            tf[r] /= len(d)
    df = (tf > 0).sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    m = tf * idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms, vocab


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 42) -> np.ndarray:
    """Seeded Lloyd's K-Means (k-means++ init). Returns a label per row."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ seeding
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() or 1.0)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        new = dists.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                C[j] = pts.mean(axis=0)
    return labels


def mine(xlsx_path: str = DEFAULT_CORRECTIONS, k: int | None = None) -> dict:
    if not os.path.exists(xlsx_path):
        print(f"No corrections file at {xlsx_path} — export some reviewed rows first.")
        return {"clusters": [], "n": 0}

    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="Review", engine="openpyxl", dtype=str).fillna("")
    mask = df["status"].astype(str).str.strip().str.lower().isin(_TARGET_STATUSES)
    rows = df[mask]
    n = len(rows)
    if n < 2:
        print(f"Only {n} rejected/needs_revision row(s) — not enough to cluster yet.")
        return {"clusters": [], "n": n}

    terms = rows["term"].astype(str).tolist()
    docs = [_tokens(f"{r.term} {r.definition} {r.comment}") for r in rows.itertuples()]
    X, vocab = _tfidf(docs)

    k = k or min(3, n)
    k = max(1, min(k, n))
    labels = _kmeans(X, k)

    clusters = []
    for j in range(k):
        members = [i for i in range(n) if labels[i] == j]
        if not members:
            continue
        centroid = X[members].mean(axis=0)
        top_idx = np.argsort(centroid)[::-1][:5]
        top_tokens = [vocab[i] for i in top_idx if centroid[i] > 0]
        clusters.append({
            "cluster": j,
            "size": len(members),
            "terms": [terms[i] for i in members],
            "top_tokens": top_tokens,
            "failure_mode_hint": ", ".join(top_tokens) or "(sparse)",
        })

    clusters.sort(key=lambda c: -c["size"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"corrections_{run}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n": n, "k": k, "clusters": clusters}, f, ensure_ascii=False, indent=2)

    print(f"\n=== Correction mining (K-Means, n={n}, k={k}) ===")
    for c in clusters:
        print(f"  cluster {c['cluster']} (size {c['size']}): {c['failure_mode_hint']}")
        print(f"      terms: {', '.join(c['terms'][:8])}")
    print(f"  -> {out_path}")
    return {"n": n, "k": k, "clusters": clusters, "out_path": out_path}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORRECTIONS
    kk = int(sys.argv[2]) if len(sys.argv) > 2 else None
    mine(path, kk)
