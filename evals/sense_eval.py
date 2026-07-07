"""
sense_eval.py — Eval #3 (Day-4): LLM-as-judge sense accuracy.

Question: when `enrich` disambiguates a word in context, does it pick the RIGHT
WordNet sense? Exact sense-id match is brittle (several senses can be acceptable),
so we ALSO use an LLM-as-judge that scores 0..5 whether the chosen sense fits the
sentence (SECURITY_EVAL.md / REVIEW_CHECKLIST H).

Two run modes (no crash either way):
  • API key present  → real pipeline: wordnet_lookup -> enrich (AI picks sense) ->
                       a SEPARATE judge AI call scores the fit 0..5.
  • No key (offline) → deterministic check only: the model is unavailable, so we
                       report exact-match of WordNet's most-common sense vs the gold
                       (a floor baseline) and mark judge scores as offline.

Usage:  python evals/sense_eval.py [golden.json]
Output: prints a table + writes evals/results/sense_eval_<run>.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from wordnet_lookup import wordnet_lookup

GOLDEN = os.path.join(_ROOT, "evals", "golden_senses.json")
RESULTS_DIR = os.path.join(_ROOT, "evals", "results")

_JUDGE_SYSTEM = (
    "You are a strict lexicography judge. Given a sentence, a target word, the WordNet "
    "sense the system CHOSE (id + gloss), and the human-reference sense, score how well "
    "the chosen sense fits the word's meaning IN THAT SENTENCE. "
    'Return STRICT JSON: {"score": <0-5 integer>, "reason": "<short>"}. '
    "5 = perfect fit, 3 = acceptable, 0 = wrong sense. Output ONLY the JSON."
)


def _gloss(senses: list[dict], sid: str) -> str:
    for s in senses:
        if s.get("sense_id") == sid:
            return s.get("definition", "")
    return ""


def _judge_ai(term, sentence, chosen, chosen_gloss, expected, exp_gloss) -> dict:
    from ai_client import call_ai
    prompt = (
        f"Sentence: \"{sentence}\"\nTarget word: {term}\n"
        f"CHOSEN sense: {chosen} — {chosen_gloss}\n"
        f"Reference sense: {expected} — {exp_gloss}\n"
        "Score the CHOSEN sense's fit in the sentence."
    )
    raw = call_ai(prompt, _JUDGE_SYSTEM)
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    obj = json.loads(text)
    return {"score": int(obj.get("score", 0)), "reason": str(obj.get("reason", ""))[:160]}


def _choose_sense(term, sentence, senses, has_key) -> str | None:
    """The sense the system picks. With a key, enrich's AI choice; else sense[0]."""
    if not senses:
        return None
    if not has_key:
        return senses[0]["sense_id"]                  # deterministic floor baseline
    import enrich as en
    drafts = en.enrich([{"term": term, "sentence": sentence, "senses": senses}], source="eval")
    node = drafts[0]["node"] if drafts else {}
    return node.get("sense_id") or senses[0]["sense_id"]


def run_eval(golden_path: str = GOLDEN) -> dict:
    with open(golden_path, "r", encoding="utf-8") as f:
        items = json.load(f)["items"]

    has_key = config.has_ai_key()
    rows, exact_hits, judge_scores = [], 0, []
    # Proactive throttle between items in AI mode: each item fires enrich + judge calls
    # back-to-back, so without a gap a 15-item run bursts past the free-tier per-minute
    # rate limit (RPM) and judge calls return None. Reuse the legacy batch-sleep knob.
    throttle = config.SLEEP_BETWEEN_BATCHES if has_key else 0

    for i, it in enumerate(items):
        if throttle and i:
            time.sleep(throttle)
        term, sentence, expected = it["term"], it["sentence"], it["expected_sense"]
        senses = wordnet_lookup(term)["senses"]
        chosen = _choose_sense(term, sentence, senses, has_key)
        exact = (chosen == expected)
        exact_hits += int(exact)

        if has_key and chosen:
            try:
                j = _judge_ai(term, sentence, chosen, _gloss(senses, chosen),
                              expected, _gloss(senses, expected))
            except Exception as e:
                j = {"score": None, "reason": f"judge failed: {e}"}
        else:
            j = {"score": (5 if exact else 0), "reason": "offline exact-match baseline"}
        if isinstance(j["score"], int):
            judge_scores.append(j["score"])
        rows.append({"term": term, "sentence": sentence, "expected": expected,
                     "chosen": chosen, "exact": exact, **j})

    n = len(items)
    summary = {
        "mode": "ai" if has_key else "offline",
        "n": n,
        "exact_match_accuracy": round(exact_hits / n, 3) if n else 0.0,
        "mean_judge_score_0_5": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
        "judge_pass_rate_ge3": round(sum(1 for s in judge_scores if s >= 3) / len(judge_scores), 3) if judge_scores else None,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"sense_eval_{run}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"\n=== Sense accuracy eval ({summary['mode']} mode, n={n}) ===")
    for r in rows:
        mark = "OK " if r["exact"] else "  ."
        print(f"  {mark} {r['term']:9} chose {str(r['chosen']):16} (gold {r['expected']:16}) "
              f"score={r['score']}")
    print(f"\n  exact-match accuracy : {summary['exact_match_accuracy']}")
    print(f"  mean judge score 0-5 : {summary['mean_judge_score_0_5']}")
    print(f"  judge pass-rate (>=3): {summary['judge_pass_rate_ge3']}")
    print(f"  -> {out_path}")
    return {"summary": summary, "rows": rows, "out_path": out_path}


if __name__ == "__main__":
    run_eval(sys.argv[1] if len(sys.argv) > 1 else GOLDEN)
