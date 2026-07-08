# DESIGN.md — Technical design (narrative)
> Spec-Driven (Day 5): `course_md/Day_5_v3.md`. Code is disposable; this + Gherkin + schema are the durable spec.

## What we build
**VocabGraph-Agent** — an agent that learns alongside a learner: from their video/audio (BYOD), it mines vocabulary into a **personal knowledge graph that grows over time**, generates Anki cards, and answers/expands questions about what was learned. Capstone track: **Agents for Good (Education)**.

## Architecture (Agent = Model + Harness — Day 1)
- **Tools (10):** see `docs/TOOLS.md`.
- **Agent:** bounded LLM tool-calling loop; reads `AGENTS.md` + the 2 `skills/*/SKILL.md`; chooses tools per intent (not hardcoded). The AI agent lives in the **interaction layer** (ask/expand/recall); ingest is a deterministic pipeline.
- **Harness:** `AGENTS.md` (rules) · `PersonalGraph` JSON (memory, grows) · in-app review table (HITL, `pending_drafts.json`) · guardrails (schema validation, tool-call cap) · observability (trajectory log).

## Data flow
```
Path B (immersion): video/audio --Whisper--> transcript --extract_vocab(AI)--> {term,sentence}
Path A (query):     user asks a word/topic ------------------------------------>
        for each term: recall -> wordnet_lookup -> enrich(AI: sense + uncertain fields, flagged)
        -> Node (validated by schema.py) -> PersonalGraph.upsert (merge=grow)
        -> build_render_graph (pyvis) + make_anki (.apkg)
        -> HITL in-app review table -> commit (app.commit_approved)
```

## Key decisions (why)
- **Deterministic-first:** WordNet gives certain edges; AI only disambiguates sense + fills uncertain fields (flagged in `source_map`). Fights hallucination (Day 4).
- **Personal graph grows:** identity by `key=lemma#sense`; new encounters append `Occurrence`. Recall is associative (peripheral hits too).
- **HITL:** nothing irreversible without human approval at the in-app review table.
- **Reuse:** ingest/extract/ffmpeg/ai_client from the existing AI-Teaching tool (tested).
- **Fallbacks:** agent-loop → button intents; graph-grows → per-film graph; both keep a working product.

## User interface (where the user talks to the agent) — GĐ4, not built yet
Two surfaces:
1. **`app.py` (Streamlit)** — the chat/intent screen. User types an intent ("mine this video", "explain X", "expand Y") or clicks an intent button → calls `agent/loop.py` (the bounded LLM tool-calling loop). This is where "agent central" is visible.
2. **In-app review table (HITL, `st.data_editor` over `pending_drafts.json`)** — the approval surface. Agent drafts units/cards → user reviews/edits/approves here → only then committed to graph/deck.
Fallback (if agent loop unstable): replace free chat with a few intent BUTTONS in app.py (LLM still does enrich/explain).

## Scope
v1 = video/audio input, English, 2 skills. Vision = PDF/image input, multi-language, i+1, feedback loop, free chatbot.

## Execution modes for building (Day 5)
Architect (scaffold, pin versions) → Builder (match style, show diffs) → Forensic (fix via logs/evidence) → Author (keep docs in sync).
