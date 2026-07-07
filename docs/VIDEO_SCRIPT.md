# VIDEO SCRIPT — VocaSync (≤5 min, Kaggle Pitch)
> Slogan: "Sync your words into one graph. Anchor them in Anki." · Tagline: "your vocabulary's second brain". Track: Agents for Good (Education).
> Rubric sections the video MUST hit (each scored): Problem · Why Agents · Architecture · Demo · The Build.
> Record the DEMO along the "watch a film" path (Ask Agent → subtitles → graph → Anki).

## 0. Prep before recording (director's data)
- Mine + Commit BOTH films first so the graph has cross-film relations:
  `Charade_25min.srt` and `His Girl Friday (1940).srt` (+ their .mp4 for media).
- Open the graph, pick ONE pair of words with a nice `is_a` / `synonym` edge across the two
  films — that pair is the hook. Note them down.
- Dry-run the hook question 2-3 times (agent path is LLM-chosen, non-deterministic) and keep
  the best take.

## 1. HOOK (0:00–0:40) — Problem + the agent answering
- On screen: learner types in **Ask Agent**:
  "I struggle to remember and understand this word — help me."
- Agent answers (let it run FULL, do NOT truncate):
  - "You met this word in film A (main node), and it relates parent/child to <word> in film B."
  - explains the USAGE SITUATION in both films (grounded — cites source @ line).
  - adds extra interpretations + a memorable association, ends "does that make sense?"
- Voiceover: flashcards store isolated sentences; a native brain stores a network. This agent
  builds that network from the films you already watch.

## 2. GROUNDING PROOF (0:40–1:30) — subtitles + AI explanation
- Cut to the film clip playing with the YELLOW highlighted subtitle (.ass) on the exact word.
- Split-screen / cut between the clip scene and the AI's explanation of the meaning-in-context.
- Point: the explanation is grounded in the real line — labelled "In your materials" (word came
  from a transcript you ingested) vs "Beyond" (extra interpretation the AI adds on top).

## 3. GRAPH (1:30–2:30) — "check the graph to see the relations"
- Agent says it will check the graph → cut to the knowledge graph.
- Show: the two films' words connected; hover a node (tooltip: term [type] (pos) · Ⓦ/🤖/✍,
  definition, 📍 source); use the search + dropdown filter; click a node → relation name on edge.
- Voiceover: this is the growing second brain — every word remembers where you met it.

## 4. ANKI (2:30–3:15) — "use Anki to practise"
- Agent suggests practising in Anki → cut to the deck.
- Show a Dictation card (audio front, type what you hear) and a card with the definition +
  screenshot cut from the film.
- Voiceover: subs2srs-style cards, generated from the film, not hand-made.

## 5. WHY AGENTS + ARCHITECTURE (3:15–4:00)
- Diagram: Model + Harness. 10 tools; agent (LLM chooses tools) vs pipeline (deterministic).
- Line: ingestion is a fixed pipeline; the INTERACTION layer is where the agent lives —
  it chooses recall → wordnet_lookup → explain on its own.

## 6. THE BUILD (4:00–4:45)
- Deterministic-first (WordNet backbone, AI only picks sense + fills uncertain fields, flagged).
- Live self-correction: stage_for_review's two-tier grounding gate flags a fabricated example
  `ungrounded` and hands back the real transcript line — the model proposes, the transcript disposes.
- HITL: one write point (commit_approved); human holds the only pen.
- Evals: golden-set LLM-as-judge; no-crash 2-tier; keys in .env only.
- Course concepts: Agent=Model+Harness · MCP · Agent Skills · Evals+Security · Spec-Driven.

## 7. CLOSE (4:45–5:00)
- "VocaSync — sync your words into one graph, anchor them in Anki." GitHub + demo link.

## Agent feasibility note (checked S17)
The hook flow works with the current build: recall surfaces provenance ("ALREADY A LEARNED
WORD" + film/sense/who-defined), cross-film relations come from is_a/synonym edges, explain
gives the grounded 3-layer answer (now self-grounds so labels aren't guessed). No code needed
for the script — only data direction + choosing a good take. Do NOT cap the agent's answer length.
