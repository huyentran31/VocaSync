---
name: building-vocab-graph
description: |
  Turns a video/audio (or its transcript) into a personal vocabulary knowledge
  graph and Anki cards. Use when the user wants to extract vocabulary from a
  film/podcast, build or update their vocab graph from content, or generate
  study cards from media. Do NOT use for: answering questions about
  already-learned words (use expanding-vocab-knowledge), translation, or
  pronunciation coaching.
version: 1.0.0
license: CC-BY-4.0
allowed-tools: [recall, ingest_transcript, extract_vocab, wordnet_lookup, conceptnet_lookup, enrich, build_render_graph, make_anki]
metadata:
  author: huyen
---
# Building a Vocab Graph from media

## When to use
- The user supplies a video/audio file (or transcript) and wants vocabulary mined.
- The user wants their personal graph + Anki deck updated from new content.

## When NOT to use
- The user asks about a word already in their graph → use `expanding-vocab-knowledge`.
- Translation, accent/pronunciation coaching, or general tutoring.

## Workflow
1. `ingest_transcript(source)` → transcript (Whisper). [deterministic]
2. `extract_vocab(transcript)` → candidate {term, sentence, surface, tag}. [AI, schema-validated]
   - **How many to find (coverage quota — REQUIRED):** the tool gathers until it has at least
     `EXTRACT_MIN_UNIQUE` (default **8**) UNIQUE items and returns up to `max_terms` (default **20**),
     using at most `EXTRACT_MAX_CALLS` (default **3**) AI calls; transcripts over `EXTRACT_CHUNK_LINES`
     (default 500) lines are split into windows (≤`EXTRACT_MAX_CHUNKS`, default 6) and gathered
     per-window so the tail is not under-sampled. **Target 8–20 learnable items per transcript.** If a
     run returns fewer than ~8 (e.g. a sparse scene), pass a larger `max_terms`, drop the `focus`
     filter, or extract a denser section — do NOT answer from only 2–3 terms. Prefer idioms / phrasal
     verbs / collocations / slang over common function words.
   - **Lemma policy:** `term` is the base form (the AI de-conjugates verbs/phrasal verbs/collocations, singularizes nouns; idioms stay in dictionary form) so inflected variants merge to ONE node by `key = lemma#sense`; the ORIGINAL form is kept in `surface`. A deterministic WordNet-morphy pass then ENFORCES the lemma for whatever WordNet can verify (single words + WordNet phrases). Long transcripts are gathered window-by-window for even coverage.
3. For each term: `recall(term)` FIRST — if already known, attach a new Occurrence instead of refetching. **ONE term per `recall` / `wordnet_lookup` call** — never a comma-separated list (the tool would split it, but per-term calls stay precise).
4. `wordnet_lookup(term)` → all senses + edges. [deterministic, free]
   - If WordNet is SPARSE (few/no edges) or the term is OOV / modern / a multi-word phrase, ALSO call `conceptnet_lookup(term)` to supplement the life-context layer (`used_for`, `has_context`) and fill missing `part_of`. You decide whether it is worth it — it is not mandatory. ConceptNet is sense-agnostic and noisier, so its edges are always flagged for review.
5. `enrich(term, sentence, senses, cn_edges)` → ONE call: pick correct sense + fill uncertain fields (collocation, mnemonic) + propose topic/exam `tags` (from the focus) + VET which ConceptNet `cn_edges` fit the chosen sense (`keep_edges`). All flagged in `source_map`. [AI]
   - **Disambiguation policy:** assign `confidence` by rubric, not by vibe — 0.9+ only when one sense unambiguously fits; cap ≤0.5 when the usage's POS doesn't match the sense's POS; null sense (conf ≤0.3) when nothing fits. Anything <0.7 is forced to HITL review.
6. `build_render_graph(units)` → networkx + Louvain cluster + pyvis. [deterministic]
7. `make_anki(units)` → .apkg (Cloze + Basic + Dictation; screenshot+audio). [deterministic]
8. STOP at the in-app review table (HITL, `pending_drafts.json` + st.data_editor) — write into the graph/deck only after human approval.

## Tool-call cap
- Max 1 `enrich` call per batch (batch all terms). Max 8 tool calls per session before asking the user.

## Examples
- Input: "mine vocab from spirited_away_ep1.mp4 about emotions" → graph cluster {annoyed, irritated, fed_up...} + deck of Cloze/Basic cards with audio clips.
- Input: "add this podcast to my graph" → new Occurrences merged into existing nodes; new nodes for new words.

## Output format
- Units validate against `schema.py` (Node/Edge/Occurrence). Graph → pyvis HTML. Deck → `.apkg`. See `assets/` for templates.

## Anti-patterns to avoid
- Don't fabricate senses/synonyms — deterministic facts come from WordNet; AI only disambiguates + fills uncertain fields (flagged).
- Don't write into the graph/deck before the HITL checkpoint.
- Don't mine every word — focus on the target items; keep it learnable.
- Don't re-encode long clips for Anki — cut SHORT (±1 sentence), screenshot+audio.
