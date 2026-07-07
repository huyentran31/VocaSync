# SECURITY & EVAL plan
> Source: `course_md/Vibe Coding Agent Security and Evaluation_Day_4 (1).md`. Lightweight, solo-feasible. Goal: credible Day-4 coverage WITHOUT enterprise overkill.

## Security (lightweight checklist)
- [ ] **No keys in code** — all secrets in `.env` (gitignored), `.env.example` documents names. (Rubric penalizes leaked keys.)
- [ ] **Context hygiene** — mask API keys / PII before logging.
- [ ] **Anti-slopsquatting** — pin deps in `requirements.txt`; run `pip-audit`; do not install hallucinated package names.
- [ ] **JIT / least privilege** — agent reads/writes ONLY inside project dir (deny-by-default); read tools stay read-only.
- [ ] **Sandbox** (if agent ever runs generated code) — subprocess + timeout + no network.
- [ ] **BYOD framing** — user supplies their OWN local video/audio → avoids copyright redistribution; the public demo must NOT redistribute copyrighted clips.

### Skipped (overkill for solo 13-day — say so in writeup)
Red/Blue/Green agent teams · AgBOM real-time · hardware MFA · full 7-pillar · cross-tenant poisoning.

## Evals ("no eval = still vibe coding" — Day 1/4)
- [ ] **Tests (deterministic):** schema validity, no orphan graph nodes, `normalize_timestamp`, merge/recall on PersonalGraph.
- [ ] **Trajectory log:** JSON per tool-call (tool, args, result, timestamp) → inspect a few sessions; detect "fragile success".
- [ ] **LLM-as-judge (intent):** golden set ~10-20 words → did `enrich` pick the correct sense? score 0-5.
- [ ] ⭐ **Mining user corrections (K-Means, Day 4):** cluster the `rejected`/`needs_revision` rows from `review.xlsx` (embeddings → KMeans) → top failure modes for the next iteration. Reuses HITL = high-value, low-cost.

### Eval dimensions touched (Day 4)
Intent satisfaction · functional correctness · trajectory quality · self-repair.

## Writeup framing (score Day-4 points)
"Lightweight security harness: .env + context hygiene + pip-audit (anti-slopsquatting) + least-privilege file access + HITL gate (Vibe-Diff style). Evals: deterministic tests + trajectory logging + LLM-as-judge sense accuracy + K-Means mining of human corrections. Enterprise patterns (red/blue/green, AgBOM, hardware MFA) intentionally out of scope for a solo tool."
