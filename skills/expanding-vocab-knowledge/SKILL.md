---
name: expanding-vocab-knowledge
description: |
  Answers questions about and expands the learner's existing vocabulary graph.
  Use when the user asks to explain a word, expand a topic, find related words,
  or recall what they have already learned. Do NOT use for: initial extraction
  from new media (use building-vocab-graph), translation, or anything outside
  the personal graph + WordNet.
version: 1.0.0
license: CC-BY-4.0
allowed-tools: [recall, wordnet_lookup, conceptnet_lookup, enrich, explain, build_render_graph]
metadata:
  author: huyen
---
# Expanding & recalling vocab knowledge

## When to use
- "Explain this word / sentence."
- "Expand the topic X" / "give me related words for Y."
- "Have I learned this word before?" / "what did I learn from film Z?"

## When NOT to use
- Mining a brand-new video/audio file → use `building-vocab-graph`.
- Requests unrelated to the personal graph or WordNet.

## Workflow
1. ALWAYS `recall(lemma)` FIRST — search main nodes + edge targets + occurrence sentences + collocations (associative, not exact-match).
   - **Looking up MANY terms? BATCH them.** Pass the whole comma-separated list in ONE `recall` call and ONE `wordnet_lookup` call — each tool splits the list and returns per-term results under `batch`. This saves the tool-call budget for the `explain` chain; do NOT spend one call per term. Use a single-term call only when looking up just one word. `explain` covers at most 5 terms per call (a token-budget cap, not the whole job — see step 3 for covering more).
   - Found → show existing graph neighborhood + past clips/cards/occurrences. Offer to add the new context.
   - Not found → proceed.
2. Gather the facts FIRST, in this ORDER:
   - `wordnet_lookup` on the concept (senses/siblings/hypernyms) — the ontology backbone, ALWAYS before ConceptNet.
   - THEN, only if WordNet was sparse/OOV or the user asks about real-world use/idioms/phrases, `conceptnet_lookup` for `used_for`/`has_context`/`part_of`.
   - optionally `enrich` new words (which vets the ConceptNet edges).
3. Call `explain` LAST — compose the answer once recall + the lookups above have gathered the facts. NEVER call `explain` before the lookups it should be grounded on, and do not keep looking things up after it.
   - **MANDATORY for meaning questions:** any "what does X mean / define / explain / giải nghĩa" request MUST be answered by calling `explain` — never answer a meaning question from your own words. `explain` is what emits the grounded layered answer: **From your graph** / **From this video: <name>** / **Dictionary** / **Beyond this video (general knowledge)**. Put its full text VERBATIM in your final answer.
   - **COVER EVERY term the learner asked about — never stop at 5.** When they ask to explain N terms (e.g. "explain all 10 phrases"), `explain` caps at 5 per call, so you MUST CHAIN calls: split the N terms into batches of ≤5 and call `explain` once per batch, looping until ALL N are covered. Do NOT emit `final` after the first batch — keep going until every term has an explanation, then concatenate ALL the `explain` outputs (in order, verbatim) into one answer. Tell the learner up front, e.g. "Explaining 10 phrases in 2 parts". If the tool-call cap is hit before you finish, say exactly which terms remain and ask to continue.
4. visualize → `build_render_graph` (only if asked).
5. If the request is ambiguous, ASK the user (do not guess).
6. New items still pass HITL (in-app **Review & Approve** table) before being committed to the graph.
7. Keeping a word: while discussing a word, if the learner signals they want to keep it
   ("save this", "add that to my list", "I want to remember this"), FIRST ASK
   *"Save X to your review list?"*. Only after they confirm, call `stage_for_review`
   (terms=[the word], source=the film/topic). It appends to the review queue — it does NOT
   commit to the graph (that still happens at Commit-Approved). Never stage unprompted.

## Tool-call cap
- Bounded per turn by the runtime cap; if you hit it, summarize and ask whether to continue.

## Examples
- Input: "explain 'fed up'" → recall (seen as synonym of 'annoyed' last week) → explain → show the connection.
- Input: "expand 'emission'" → recall → wordnet_lookup neighbors → add fresh related nodes (flagged) → updated graph.

## Output format
- Answers cite prior occurrences (source + sentence) when recall hits. New nodes/edges validate against `schema.py`.

## Anti-patterns to avoid
- Don't refetch from WordNet/AI if `recall` already has the answer — reuse memory.
- Don't invent relations; WordNet first, AI flagged.
- Don't exceed the tool-call cap; ask instead of looping.
