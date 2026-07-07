"""
explain.py — Tool #8 (read, medium cost, AI).

Explain a word / sentence / grammar point for an English learner, in plain language.
This is an interaction-layer tool: the agent typically calls recall() first and passes
the hits in as `context` so the answer is GROUNDED in what the learner already saw
(cites prior source + sentence) instead of a generic dictionary blurb.

The legacy Gemini client forces a JSON response mime type, so we ask the model for
{"explanation": "...markdown..."} and unwrap it (falling back to raw text if needed).

Error model: missing key → SystemError_ (HALT); transient AI/parse failure → retry
once, then return a clear fallback string (explain must not crash a turn).
"""

from __future__ import annotations

import json
import re

import config
from ai_client import call_ai
from _common import SystemError_, log_tool_call

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Word, phrase, sentence, or grammar question."},
        "context": {"description": "Optional recall() hits to ground the answer."},
    },
    "required": ["query"],
}

_SYSTEM_PROMPT = (
    "You are a friendly English teacher. Explain the learner's query clearly: meaning, when to "
    "use it, and natural example(s). Answer completely — do not cut it short.\n"
    "Enrich the answer with whatever is USEFUL for THIS word/phrase in context — you decide "
    "which of these to include (add the relevant ones, skip what does not fit; do NOT force all, "
    "keep it focused not exhaustive): 2+ example sentences in different contexts; key synonyms / "
    "antonyms; common collocations or set phrases; common mistakes / easily-confused words; "
    "register (formal / informal / slang); pronunciation (IPA). Draw synonyms/relations from the "
    "Dictionary layer when available; anything from your own knowledge goes in the 'Beyond this "
    "video' layer below.\n"
    "Be explicit about WHERE each fact comes from, using up to THREE provenance layers (include "
    "only the layers you actually have; label them so the learner can tell them apart):\n"
    "1) **From your graph** — ONLY if the provided context shows prior occurrences "
    "(source @ timestamp) or related words the learner already saved: weave them in naturally "
    "('you saw this in Charade @ 03:12; a related word you saved is annoyed (synonym)'). Use only "
    "the sources/timestamps/related words that appear in the context — NEVER invent them; use the "
    "source file NAME, not a disk path. If the context has none of this, DO NOT claim anything is "
    "from the learner's graph. Prior occurrences from the learner's SAVED GRAPH (earlier films/"
    "sessions) belong ONLY in this layer — even when they name a film, NEVER put them under a "
    "'From this video' heading (that heading is reserved for the video currently being watched).\n"
    "1b) **From this video: <video name>** — use this heading ONLY when the context has a "
    "'FROM THIS VIDEO' block (the video the learner is watching NOW); without that block this "
    "layer must not appear at all. When present, it "
    "lists the queried terms that ACTUALLY appear in the video the learner is watching right now, "
    "each with its real transcript line. Head this section 'From this video: <name>' using the "
    "video's name from the block (e.g. 'From this video: Charade'). For each queried term, state "
    "whether it is IN THIS VIDEO (quote its line VERBATIM from the block) or NOT — when the "
    "block is present, EVERY queried term gets this section: either its verbatim line, or "
    "exactly 'Not found in this video.' (so the learner knows the phrase is extra). Do NOT "
    "substitute a 'similar' line, do NOT paraphrase the plot, do NOT alter a quoted line — a "
    "wrong or lookalike quote is worse than none (put any extra knowledge in layer 3 instead). "
    "The system VERIFIES every quote in this section against the real transcript and replaces "
    "any line that is not verbatim, so never guess or reconstruct a line — quote only from the block.\n"
    "2) **Dictionary** — definitions, senses and relations "
    "(synonym/antonym/is_a/part_of/used_for) are the deterministic reference backbone; present "
    "them as the core of the explanation. This is legitimately grounded knowledge (not invented).\n"
    "3) **Beyond this video (general knowledge)** — anything from your own understanding "
    "(register, nuance, origin, cultural notes) goes in its own clearly labelled section.\n"
    "Keep the layers visually distinct so the learner always knows which facts come from THIS "
    "video, which are dictionary-grounded, which come from their own saved graph, and which are "
    "general model knowledge. Give each term a COMPLETE, self-contained entry — you are covering "
    "only a few terms in ONE call, so be THOROUGH, not thin. For EVERY term include: its meaning; "
    "its video line (if any); at least 2 ORIGINAL example sentences that YOU compose in DIFFERENT "
    "everyday contexts — NEVER reuse the video/transcript line as an example, and do not repeat "
    "the same example twice; and the relevant "
    "extras — key synonyms/antonyms, common collocations, a common mistake or easily-confused "
    "word, register, and pronunciation (IPA). Keep ALL of one term's info together under that term "
    "(do NOT lump register/pronunciation for every term into one shared block at the very end).\n"
    "FORMATTING (important): format each term as its OWN markdown block — a heading line "
    "'### <term>', then EACH field (Meaning, From this video, Examples, Synonyms, Register, "
    "Pronunciation, …) on its OWN line as a bold label (e.g. '**Meaning:** …'), examples as a "
    "bulleted list. Use real line breaks (\\n) between fields — NEVER run a term's fields together "
    "into one long paragraph. Put a blank line between terms.\n"
    'Return STRICT JSON: {"explanation": "<markdown text>"}. Output ONLY that JSON. '
    "The markdown value MUST use \\n for line breaks so it renders as separate lines."
)


def _context_str(context) -> str:
    """Render recall() hits into a compact grounding block (or '')."""
    if not context:
        return ""
    try:
        import os
        # S17-5.1: a batch recall result ({"batch": {term: hits}}) -> render each term's hits.
        batch = context.get("batch") if isinstance(context, dict) else None
        if isinstance(batch, dict) and batch:
            blocks = [f"[{t}]\n{g}" for t, g in
                      ((t, _context_str(r)) for t, r in batch.items()) if g]
            return "\n".join(blocks)
        lines = []
        main = context.get("as_main_node") if isinstance(context, dict) else None
        if main:
            lines.append(f"Known word: {main.get('term')} ({main.get('category','')})")
            # The node's OWN provenance (S17-5.1c): saved definition + who authored it +
            # where the learner met the word — this is what grounds "From your graph".
            if main.get("definition"):
                by = (main.get("source_map") or {}).get("definition", "")
                lines.append(f"Saved definition ({by or 'unknown source'}): {main['definition']}")
            for occ in (main.get("occurrences") or [])[:4]:
                if not isinstance(occ, dict):
                    continue
                src = os.path.basename(str(occ.get("source", "") or "?"))
                start = str(occ.get("start", "") or "").strip()
                where = f"{src} @ {start}" if start else src
                lines.append(f'- learner met it in {where}: "{occ.get("sentence", "")}"')
            # Related learned words: the target + relation type of each edge already in the
            # graph (synonym/antonym/is_a...). Only what is in context — never invented. Cap ~6.
            rel_edges = []
            for e in (main.get("edges") or [])[:6]:
                tgt = str(e.get("target", "")).split("#")[0].strip()
                rtype = str(e.get("type", "")).strip()
                if tgt:
                    rel_edges.append(f"{tgt} ({rtype})" if rtype else tgt)
            if rel_edges:
                lines.append("Related learned words: " + ", ".join(rel_edges))
        # Prior occurrences: WHERE (source file name) and, when timed, @ the timestamp.
        for h in (context.get("in_sentences", []) if isinstance(context, dict) else [])[:4]:
            src = os.path.basename(str(h.get("source", "") or "?"))
            start = str(h.get("start", "") or "").strip()
            where = f"{src} @ {start}" if start else src
            lines.append(f'- seen in {where}: "{h.get("sentence","")}"')
        rel = context.get("as_related", []) if isinstance(context, dict) else []
        if rel:
            lines.append("Related nodes: " + ", ".join(rel[:6]))
        return "\n".join(lines)
    except Exception:
        return ""


def _materials_str(context) -> str:
    """S18 #6: render the 'IN YOUR MATERIALS' grounding block — the terms that actually appear
    in the transcript the learner is mining right now, with their real source line. This is the
    layer that lets explain say "this phrase is FROM the film" vs "this is extra knowledge".
    Shape: context['materials'] = {'source': <film name>, 'hits': [{'term','line','start'?}, ...]}.
    Returns '' when there is no materials context (degrade — no false 'from the film' claim)."""
    if not isinstance(context, dict):
        return ""
    mat = context.get("materials")
    if not isinstance(mat, dict):
        return ""
    hits = [h for h in (mat.get("hits") or []) if isinstance(h, dict) and h.get("line")]
    if not hits:
        return ""
    src = str(mat.get("source", "") or "the transcript")
    lines = [f"FROM THIS VIDEO — {src} (queried terms that appear in it, with their verbatim line):"]
    for h in hits[:10]:
        term = str(h.get("term", "") or "").strip()
        start = str(h.get("start", "") or "").strip()
        where = f"{src} @ {start}" if start else src
        lines.append(f'- "{term}" appears in {where}: "{h.get("line")}"')
        # FEATURE-3 (lazy): the ±N neighbouring lines (a whole exchange), present only when the
        # learner asked for the in-scene meaning — gives the model the surrounding moment to read
        # nuance from. scene_before/after are LISTS (oldest→newest); the queried line is marked [].
        def _seq(v):
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
            return [str(v).strip()] if str(v or "").strip() else []
        sb, sa = _seq(h.get("scene_before")), _seq(h.get("scene_after"))
        if sb or sa:
            around = " … ".join(sb + [f"[{h.get('line')}]"] + sa)
            lines.append(f'    scene context: {around}')
    return "\n".join(lines)


def _unwrap(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ("explanation", "text", "answer"):
                if isinstance(obj.get(k), str):
                    return obj[k].strip()
        if isinstance(obj, str):
            return obj.strip()
    except Exception:
        pass
    # SALVAGE a malformed {"explanation": "..."} envelope. The model routinely leaves the
    # example sentences' inner quotes unescaped ("Please go on...") and sometimes adds a
    # stray trailing brace, so json.loads fails and the learner saw the raw JSON dumped.
    # Pull the value between the first `"explanation":"` and the final `"` (greedy, so inner
    # quotes stay as content), then decode the backslash escapes in ONE left-to-right pass.
    m = re.search(r'"(?:explanation|text|answer)"\s*:\s*"(.*)"[\s}]*$', text, re.DOTALL)
    if m:
        esc = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
        return re.sub(r"\\(.)", lambda mm: esc.get(mm.group(1), mm.group(1)),
                      m.group(1)).strip()
    return text  # not JSON — return raw text as-is


_RECALL_KEYS = ("as_main_node", "as_related", "in_sentences", "in_collocations")


def _self_ground(query: str) -> str:
    """S17-5.1c: when the agent passes NO usable recall context, ground the answer
    ourselves — recall() each comma-separated term of the query (capped at 5, same
    limit the system prompt states) and render the hits. Without this, the
    'From your graph / Dictionary / Beyond' labels were the model's guess (vibes),
    not evidence. Deterministic, read-only; failure degrades to '' (ungrounded)."""
    try:
        from recall import recall as _recall
        terms = [t.strip() for t in str(query or "").split(",") if t.strip()][:5]
        blocks = []
        for t in terms:
            g = _context_str(_recall(t))
            if g:
                blocks.append(g if len(terms) == 1 else f"[{t}]\n{g}")
        return "\n".join(blocks)
    except Exception:
        return ""


def explain(query: str, context=None) -> str:
    """Return a learner-friendly explanation string for `query`."""
    args = {"query": query, "has_context": bool(context)}
    if not (query or "").strip():
        return ""
    if not config.has_ai_key():
        msg = "No AI API key configured (set GEMINI_API_KEY/AI_API_KEY in .env)."
        log_tool_call("explain", args, error=msg)
        raise SystemError_(msg)

    # Agent-path grounding (S17-5.1c): the pipeline always passes real recall() hits;
    # the agent sometimes passes nothing or a non-recall blob. If the context is not
    # recall-shaped, fetch the learner's real graph hits ourselves so the provenance
    # labels in the answer are evidence-based.
    if not (isinstance(context, dict) and any(k in context for k in _RECALL_KEYS)):
        grounding = _self_ground(query)
        args["self_grounded"] = bool(grounding)
    else:
        grounding = _context_str(context)
    # S18 #6: prepend the "in your materials" block (terms found in the current transcript with
    # their real line) so the answer separates FILM vocab from extra knowledge. Additive: absent
    # materials -> "" -> unchanged behaviour.
    materials = _materials_str(context)
    args["has_materials"] = bool(materials)
    grounding = "\n".join(b for b in (materials, grounding) if b)
    # FEATURE-3 (lazy): when the learner asked for the in-scene meaning, add a directive so the
    # model interprets the word AS USED in this specific scene (using the 'scene context' lines)
    # — kept separate from the general dictionary meaning. Absent flag -> unchanged behaviour.
    in_scene = isinstance(context, dict) and bool(context.get("explain_in_scene"))
    args["in_scene"] = in_scene
    scene_note = (
        "\nThe learner asked about the meaning IN THIS SCENE. For each term that appears in the "
        "video, add a '**In this scene:**' line explaining what the term means and what the "
        "character is doing/feeling AS USED HERE (read the 'scene context' lines around the "
        "quoted line), and WHY it is phrased this way — keep this distinct from the general "
        "Dictionary meaning.\n"
    ) if in_scene else ""
    user_prompt = (f"Prior context (the learner has seen these):\n{grounding}\n\n" if grounding else "") \
        + scene_note + f"Explain: {query}"

    for attempt in range(2):
        try:
            raw = call_ai(user_prompt, _SYSTEM_PROMPT)
            out = _unwrap(raw)
            if out:
                log_tool_call("explain", args, result={"chars": len(out)})
                return out
        except Exception as e:
            if attempt == 1:
                msg = f"explain: AI call failed: {e}"
                log_tool_call("explain", args, error=msg)
                return f"(Could not generate an explanation right now: {e})"
    log_tool_call("explain", args, result={"chars": 0})
    return "(No explanation produced.)"


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "fed up"
    print(explain(q))
