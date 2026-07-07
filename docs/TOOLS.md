# TOOLS.md — Tool contract (8 tools)
> Standard source: `course_md/Agent Tools & Interoperability_Day_2.md` (MCP, tool definition).
> Rule (Day 2): each tool = description (for LLM routing) + JSON schema in/out (from `schema.py`) + scope + cost + error behavior. Python function = source of truth; MCP server = thin wrapper (transport: **stdio**).

| # | Tool | Description (routing) | Input → Output | Type | Scope | Cost | On error |
|---|---|---|---|---|---|---|---|
| 1 | `recall(lemma)` | Find every trace of a word in the personal graph (main node / edge target / sentence / collocation), associative not exact. | `str` → `dict` hits | Python | read | cheap | return empty hits |
| 2 | `ingest_transcript(source)` | Transcribe a video/audio file to text+timestamps. | path → `srt/segments` | Python (Whisper, **reuse old tool**) | read | medium | system-error if file/Whisper missing |
| 3 | `extract_vocab(transcript, focus?)` | Find candidate {term, sentence} from transcript. | text → `list[dict]` | AI (**reuse old `ai_client`**) | read | medium | clip-error per bad item; validate schema, retry |
| 4 | `wordnet_lookup(term)` | Return ALL WordNet senses + edges (syn/ant/is_a/hyponym/category). | `str` → `list[sense]` | Python (NLTK) | read | cheap | return [] if not found |
| 5 | `enrich(term, sentence, senses)` | ONE call: pick correct sense + fill uncertain fields (collocation, mnemonic), flag in `source_map`. | → `Node` | AI | write(draft) | **expensive** (batch!) | retry on invalid schema; flag low-confidence |
| 6 | `build_render_graph(units)` | networkx graph + Louvain cluster + pyvis HTML. | `list[Node]` → html path | Python | write(local) | cheap | skip bad node, log |
| 7 | `make_anki(units)` | .apkg: Cloze + Basic + Dictation; screenshot+audio; stable GUID. | `list[Node]` → .apkg path | Python (genanki + **reuse ffmpeg**) | write(local) | medium | clip-error per card; continue |
| 8 | `explain(query)` | Explain a word/sentence/grammar point. | `str` → `str` | AI | read | medium | retry |

## MCP notes (Day 2)
- Transport **stdio** (JSON-RPC 2.0) — simplest for capstone.
- Each tool's input/output schema derived from `schema.py` (Pydantic → JSON schema). Pydantic validation = the anti-hallucination barrier.
- read tools (recall/wordnet/explain) → read-only. write tools (ingest/enrich/build/make_anki) → scoped to project dir only.
- Log every tool call (audit + trajectory eval — see `docs/SECURITY_EVAL.md`).
- `.env` for any API keys; never hardcode (Day 4).

## Agent routing (Day 1/3)
The LLM (bounded loop) reads these descriptions + the 2 SKILL.md to choose tools per intent. NOT a hardcoded if/else. Cap 5-8 tool calls/turn; ambiguous → ASK.
