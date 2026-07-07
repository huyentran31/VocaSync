"""
registry.py — the single source of truth for the 10 tools.

The Python functions ARE the tools (docs/TOOLS.md, Day-2): the agent loop, the MCP
server, and app.py all dispatch through this one catalog so a tool is described,
scoped, and validated in exactly one place. Each entry carries the routing
description the LLM reads + the JSON input schema + read/write scope.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recall import recall, INPUT_SCHEMA as RECALL_IN
from ingest_transcript import ingest_transcript, INPUT_SCHEMA as INGEST_IN
from extract_vocab import extract_vocab, INPUT_SCHEMA as EXTRACT_IN
from wordnet_lookup import wordnet_lookup, INPUT_SCHEMA as WN_IN
from conceptnet_lookup import conceptnet_lookup, INPUT_SCHEMA as CN_IN
from enrich import enrich, INPUT_SCHEMA as ENRICH_IN
from build_render_graph import build_render_graph, INPUT_SCHEMA as GRAPH_IN
from make_anki import make_anki, INPUT_SCHEMA as ANKI_IN
from explain import explain, INPUT_SCHEMA as EXPLAIN_IN
from stage_for_review import stage_for_review, INPUT_SCHEMA as STAGE_IN


# name -> {fn, description (LLM routing), schema, scope}
TOOLS: dict[str, dict] = {
    "recall": {
        "fn": recall, "schema": RECALL_IN, "scope": "read",
        "description": "Find every trace of a word in the personal graph (main node / "
                       "edge target / sentence / collocation), associative not exact. Call FIRST.",
    },
    "ingest_transcript": {
        "fn": ingest_transcript, "schema": INGEST_IN, "scope": "read",
        "description": "Transcribe a video/audio file (or parse an .srt) into timestamped segments.",
    },
    "extract_vocab": {
        "fn": extract_vocab, "schema": EXTRACT_IN, "scope": "read",
        "description": "From a transcript, select candidate {term, sentence} learning items.",
    },
    "wordnet_lookup": {
        "fn": wordnet_lookup, "schema": WN_IN, "scope": "read",
        "description": "The ontology backbone — return ALL WordNet senses + grounded edges "
                       "(syn/ant/is_a/hyponym/part_of/category). Always call this BEFORE conceptnet_lookup.",
    },
    "conceptnet_lookup": {
        "fn": conceptnet_lookup, "schema": CN_IN, "scope": "read",
        "description": "Supplement WordNet with ConceptNet life-context edges "
                       "(part_of/used_for/has_context). Call ONLY AFTER wordnet_lookup, and only if "
                       "WordNet was sparse/missing those relations or the term is OOV — never before "
                       "wordnet_lookup. Edges are flagged for review.",
    },
    "enrich": {
        "fn": enrich, "schema": ENRICH_IN, "scope": "write_draft",
        "description": "ONE AI call: pick the correct sense + fill uncertain fields "
                       "(collocations, mnemonic), flagged for review. Batch all terms.",
    },
    "build_render_graph": {
        "fn": build_render_graph, "schema": GRAPH_IN, "scope": "write_local",
        "description": "Render vocab nodes into a clustered (Louvain) pyvis HTML graph.",
    },
    "make_anki": {
        "fn": make_anki, "schema": ANKI_IN, "scope": "write_local",
        "description": "Build an Anki .apkg (Cloze + Basic + Dictation; short audio + screenshot).",
    },
    "explain": {
        "fn": explain, "schema": EXPLAIN_IN, "scope": "read",
        "description": "Compose the FINAL learner-facing answer for a word / sentence / grammar point. "
                       "Call this LAST — only after recall and any wordnet/conceptnet lookups have "
                       "gathered the facts to ground it on.",
    },
    "stage_for_review": {
        "fn": stage_for_review, "schema": STAGE_IN, "scope": "write_draft",
        "description": "Save specific words the learner asked to keep into their review queue for "
                       "later approval. Does NOT commit to the graph. Ask the learner first; only "
                       "stage words they explicitly want to keep.",
    },
}


def tool_catalog() -> str:
    """Human/LLM-readable catalog (name — description — args) for the system prompt."""
    lines = []
    for name, t in TOOLS.items():
        props = (t["schema"].get("properties") or {})
        req = set(t["schema"].get("required") or [])
        arglist = ", ".join(
            (f"{k}*" if k in req else k) for k in props
        )
        lines.append(f"- {name}({arglist}) [{t['scope']}] — {t['description']}")
    return "\n".join(lines)


def call_tool(name: str, args: dict):
    """Dispatch by name with kwargs. Raises KeyError for an unknown tool.

    Zero-trust (Day 5): every call first passes the structural Policy gate
    (policy.py / execution_policy.yaml) before the tool can touch disk or network.
    """
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name}")
    from policy import default_policy
    default_policy().check(name)          # PolicyViolation if role/env forbids it
    return TOOLS[name]["fn"](**(args or {}))
