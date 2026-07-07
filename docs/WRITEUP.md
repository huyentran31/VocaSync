# VocaSync — a vocabulary agent that grows a personal knowledge graph from the media you watch

> **"Sync your words into one graph. Anchor them in Anki."** — *your vocabulary's second brain*
> Track: **Agents for Good (Education)** · Kaggle × Google Capstone
> [SCREENSHOT: cover — app main view with a grown graph]

## 1. The problem

I learned English the way most self-studiers do: watch a film, meet a phrase, put it on a flashcard. Before this project I was maintaining a 20-column spreadsheet by hand for my HSK vocabulary. It taught me the real problem: **flashcard tools store sentences one at a time, but a native speaker's memory stores a network.** "Fed up" lives next to "annoyed", under "emotions", inside that scene where you first heard it. When your tools store isolated cards, you review more and remember less, because nothing connects.

Research on semantic interference backs this up: unrelated items reviewed together compete, while items recalled through their relations reinforce each other. The fix is not a better flashcard. It is a **graph**.

## 2. The solution

VocaSync is a learning companion that turns the media you already watch into a **personal vocabulary knowledge graph that grows across sessions**, plus an Anki deck cut straight from the film. You attach the video (a subtitle file is optional — if you don't have one, Whisper transcribes the audio for you), press **Mine**, review what the AI proposes, and commit. Every committed word remembers *where you met it* — film, sentence, timestamp — and *how it relates* to everything else you know.

Then you talk to it. "Have I learned 'go on'?" The agent answers from your graph: which film, which line, who wrote the definition you saved. This is immersion tooling, not exam-prep tooling: the graph is yours, grown one film at a time.

This targets a real, hard workflow. Serious sentence-miners (the Anki/immersion crowd) build the gold-standard card by hand — screenshot the frame, rip the sentence audio, copy the subtitle line, paste each into the right Anki field — **two to five minutes of clicking per card**, with clip audio that drifts when trimmed by ear. VocaSync collapses that: timestamps come straight from the subtitle cue (the AI never touches timing, so a cut is exact and deterministic), ffmpeg cuts the audio and a mid-frame screenshot, and a Dictation card (type what you hear) is generated when audio is available — the subs2srs recipe, produced instead of assembled. The miner's job shrinks back to the only part worth a human: deciding which words are worth keeping.

[SCREENSHOT: knowledge graph with relation chips + tooltip showing Ⓦ/🤖/✍ meaning-source badge]

## 3. Why an agent, and not just a pipeline

Both — and the distinction is the architecture.

The ingestion path (film → candidates → enrichment → review) is a **deterministic pipeline**: the same tool sequence every time, because extraction should be repeatable, testable, and cheap to reason about.

The interaction layer is where the **agent** lives. When a learner asks "explain these phrases and tell me which ones I've seen before", no fixed sequence fits: the LLM must `recall` each term from the graph, look up senses in WordNet, and only then compose an `explain` — chaining tool calls it chooses itself, up to a per-turn cap. A real trajectory:

```
Agent chose 5 tool(s): recall → recall → wordnet_lookup → wordnet_lookup → explain
```

The agent reads two SKILL.md files plus the tool catalog and picks its own route; the pipeline never asks the model what to do next. The UI makes the split visible: the **Ask Agent** input runs the agent, while the **Mine / Explain / Expand** actions run fixed pipelines. [VERIFY UI vs screenshot]

## 4. Architecture: Model + Harness

```
TOOLS (10)  recall · ingest_transcript · extract_vocab · wordnet_lookup ·
            conceptnet_lookup · enrich · build_render_graph · make_anki ·
            explain · stage_for_review
AGENT       LLM tool-calling loop — bounded at 10 tool calls/turn, asks when ambiguous,
            read-only against the graph (its only write tool is stage_for_review,
            which writes to the REVIEW QUEUE, never the graph)
HARNESS     AGENTS.md (operating contract) · 2 Agent Skills (SKILL.md) ·
            PersonalGraph JSON (persistent memory) · Pydantic schema (data contract) ·
            in-app HITL review · .env (secrets)
```

[SCREENSHOT: architecture diagram]

Three design rules carry the whole system:

**Deterministic first.** WordNet (local) is the backbone: senses, synonyms, antonyms, hypernyms all come from it as verifiable edges. The AI's job is narrow — pick which sense fits the sentence, and fill the fields a dictionary cannot (collocations, mnemonics, topic tags). Every AI-authored field is marked `source_map='ai'` and surfaced in review. (ConceptNet is an *optional* online lookup that supplements WordNet with life-context edges; it is enabled by default and called when WordNet is insufficient. Because its public API was intermittently returning 502s, we ran with the per-term Mine calls turned off — `CONCEPTNET_PER_TERM=0` — so word meanings come from local WordNet; the agent can still call it on demand.)

**One write point.** Nothing reaches the graph except `commit_approved`, which runs only when the human presses Commit. The agent cannot write the graph at all. Final exports (deck, Obsidian vault, infolog, highlighted `.ass` subtitle) are generated *after* commit, from the approved subset only — an unreviewed word cannot leak into any deliverable.

**No-crash, two tiers.** A bad card or a failed clip is a *clip-error*: mark it, log it, continue. A missing API key or a corrupt graph is a *system-error*: halt with a clear message. When ConceptNet's public API started returning intermittent 502s mid-project, the pipeline degraded via a config knob (`CONCEPTNET_PER_TERM=0`, which skips the per-term Mine calls) instead of crashing — and a live 502 simply yields no ConceptNet edges rather than an error. The error model doing exactly what it was designed for.

## 5. How a session actually runs

A real run from this submission, on *His Girl Friday* (1940, public domain):

1. **Mine.** The subtitle is chunked into windows; each window goes through one extraction call. A Python self-correction pass then drops any candidate that does not literally appear in the transcript — the model proposes, the transcript disposes. Candidates get lemmatized (so "reduces" and "reduced" merge into one node), sense-tagged against WordNet, and enriched in a single AI call per batch.
2. **Review.** The candidates land in an editable table. Two machine columns do different jobs: **⚠ ai flag** shows *why* a row deserves a human look ("polysemy, definition(ai)" — an empty cell means no concern), while **🤖 ai fields** shows *which fields the AI authored*, even on confident rows, so machine-written content is never invisible. A WordNet sense browser fixes a wrong sense on the spot. Rows whose example sentence is not found verbatim in the transcript carry an `ungrounded` flag with the nearest real line attached, so a hallucinated example is caught before a human trusts it. [VERIFY UI vs screenshot]
3. **Commit.** A second deterministic validator re-checks every approved row — fabricated sense IDs, empty definitions, and words with no grounded source sentence are *partitioned*: held back with a per-row reason (`#5 'flyback' — no grounded occurrence in source`) while valid rows commit. Human approval is necessary but not sufficient.
4. **Artifacts.** The graph re-renders with the new words gold-ringed. The Anki deck is cut from the companion video with ffmpeg: each card gets the sentence's audio and a mid-frame screenshot. An infolog (word → source @ time → sentence) and an Obsidian vault export round it out.

In the measured run: 40 subtitle cues → 11 candidate phrases (mostly idioms and phrasal verbs — the extractor is prompted to prefer them over common words) → 11 committed → an Anki deck, one graph, 11 vault notes.

[SCREENSHOT: review table with ⚠ ai flag reasons + 🤖 ai fields columns]
[SCREENSHOT: an `ungrounded` row showing the fabricated sentence flagged + the real transcript line returned]
[SCREENSHOT: Anki card with audio, screenshot and Dictation]

## 6. Course concepts applied

| Concept (Day) | Applied as | Evidence in repo |
|---|---|---|
| Agent = Model + Harness (D1) | Bounded tool-loop under an AGENTS.md contract; agent vs pipeline split | `agent/loop.py`, `AGENTS.md` |
| MCP (D2) | stdio server exposing the full 10-tool registry | `mcp_server.py` |
| Agent Skills (D3) | Procedural memory the agent loads per task (2 skills) | `skills/*/SKILL.md` |
| Evals + Security (D4) | Golden-set LLM-as-judge; K-Means failure mining; HITL gate; key hygiene | `evals/`, `tests/` |
| Spec-Driven (D5) | Gherkin specs the code must repay (3 feature specs) | `specs/features/*.gherkin` |

One spec-driven story worth telling: `enrich.gherkin` promised early on that "an AI definition is flagged 'ai'". The code shipped without it — out-of-vocabulary words reached review with an unmarked definition. The spec sat there as recorded debt until a later session repaid it: OOV definitions now carry `source_map='ai'`, always force review, and show a 🤖 badge in the graph tooltip. The spec was the contract that made the gap visible.

## 7. Build journey — the decisions that mattered

**The AI proposes, the transcript decides.** The recurring anti-hallucination pattern is a cheap deterministic check wrapped around every model output. Extraction drops candidates not literally present in the transcript. Enrichment caps confidence at 0.3 for out-of-vocabulary terms *in code*, because the model cheerfully returns 1.0 for words WordNet has never seen. The agent's save tool (`stage_for_review`) runs a **two-tier grounding gate**: the word must appear in its claimed sentence, *and that sentence must occur in the ingested transcript*. A fabricated example is flagged `ungrounded` and the rejection returns the real transcript line to the agent, so the model re-reads the source and corrects itself — live self-correction closing inside a single turn.

**Killing Excel.** The original HITL loop was an exported review.xlsx so it's more familiar for users. It turned out less eficient in usability testing: file locks, format drift, a modal "close the file to continue" dance. The replacement is an in-app editable table backed by a JSON stash on disk. Lesson: the checkpoint matters, the medium does not.

**The comma-list bug.** In live use, the agent packed twelve phrases into one call — `recall(lemma="make it work, go on, …")` — got a silent `found=False` for the whole string, and confidently mislabeled its answer's provenance. The fix: the tools now split a comma-list and answer each term separately, and `explain` re-grounds itself by calling `recall` internally when handed no usable context. Provenance labels are now backed by an actual graph read, not the model's guess.

**A leaner review, a denser graph, cross-film links.** Later sessions were mostly subtraction: one flag column that explains itself instead of a boolean, tooltips cut to a few lines with an icon badge (Ⓦ WordNet / 🤖 AI / ✍ you) for where each definition came from. The graph view gained a pair rule — a relation filter shows an edge only with both endpoints visible — because a dangling edge is a small lie. The graph also links words *across films*: mine several movies and a phrase like **get out** becomes one node carrying two films and two senses (a command to leave in one film, "escape" in another), while **sane** in one film wires by antonym to **insane** in another. When you ask about a word you already saved elsewhere, the answer surfaces a "🔗 Connects to your graph" block. [VERIFY cross-film counts vs a fresh mine]

## 8. Results and evals

Deterministic tests: **16 test files pass** — schema, dedup/provenance, HITL gate, export gate, policy gate, partitioned commit, agent-loop cap, batch splitting, and transcript-grounding. Of these, 15 run fully offline; one (`test_conceptnet.py`) calls a live public API. The commit path is tested twice — as a mirror and by calling the real `app.commit_approved` with sandboxed side effects — because a mirror can drift from the function it imitates.

**Sense accuracy (LLM-as-judge, golden set n=15, live AI):** exact synset-ID match **40%**, but mean judge score **4.0/5** with an **80% pass rate (≥3)**. The gap is the finding: the judge credits semantically correct synonym synsets that ID-matching rejects. Failures cluster on heavily polysemous words — which is exactly why the review table's polysemy flag exists (≥5 senses → always reviewed). The eval found the weakness; the UI now guards it.

**Correction mining:** TF-IDF + K-Means (pure numpy, seeded) cleanly separates a synthetic correction set into its two planted failure modes — *wrong-sense* vs *missing-WordNet-entry*. Honest status: verified on synthetic data; the wiring that appends real human edits to a correction log at commit time is designed but awaiting a real corpus. The broader pattern is the grounding playbook at small scale: context adherence, chunk attribution (source @ timestamp), confidence enforced in code, and abstention with a human judge as the final gate.

**Security:** keys live in `.env` only, sent via header (never URL), masked in logs. The agent writes only inside the project directory; the single sanctioned outside write (copying a subtitle next to your video) happens only on an explicit button press. No dependency named "ConceptNet" was ever installed — the namespace is a known slopsquatting target; the REST API is called directly instead.

## 9. Vision & limitations

The engine generalizes; the demo is deliberately narrow. **Signature: "The AI proposes, the transcript decides." · "Your data stays local — the only network call is the LLM API."**

**Vision / roadmap** (planned, not yet built):
- **Ingest adapters for asbplayer / Language Reactor / Yomitan.** Today the ingestion tool accepts `.srt`/`.vtt`/`.txt`/`.md` and media files only. An adapter reading their multi-sense export dumps would position VocaSync as the downstream AI layer that collapses a noisy dump into the one sense that fits the scene.
- **Wider input & more languages.** The same funnel extends to PDFs, articles, and photographed pages (OCR); the graph schema is language-agnostic, so Chinese/Japanese need reading layers and a segmenter where WordNet is swapped for a language-appropriate lexicon. My hand-made HSK spreadsheet is the first migration target.
- **Optional richer meanings via ConceptNet** once the public API is stable again.
- **Closing the learning loop.** Pull review scores back from Anki (AnkiConnect) so the graph knows which words are weak and biases `expand` toward them.

**Limitations** (stated plainly — transparency is a feature):
- **Quality depends on subtitle↔video sync.** Anki audio is cut by subtitle-cue timestamp; if a `.srt` drifts from its video, the audio drifts too, so the app only exports audio/highlight for films whose subtitles match.
- **Ask Agent grounding edge case (known, tracked).** When a saved word is referred to indirectly ("this phrase") inside the "From your graph" layer, the anti-fabrication guard can substitute "Not found in this video." where a real prior line exists. A fix is in progress; asking about the word by name avoids it.
- **Lightweight LLM.** The generator sometimes ignores prompt instructions — which is *why* Python owns every verifiable decision (grounding, timestamps, summaries). That split is the design's strength, not an afterthought.
- **No in-app video player.** Review-with-highlight happens by opening the exported `.ass` subtitle in an external player (VLC / MPC).

Each roadmap item is an input or lexicon swap at the edges of the harness. The core loop — deterministic spine, agent interaction layer, a graph that is genuinely yours, and a human holding the only pen that writes to it — stays the same.

---
*GitHub: [LINK] · Demo video: [LINK] · License: CC-BY 4.0*
