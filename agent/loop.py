"""
loop.py — the bounded LLM tool-calling loop (the "agent" in VocabGraph-Agent).

The LLM CHOOSES which tool to call (Day-1/3) — this is NOT a hardcoded if/else. It
reads AGENTS.md + both SKILL.md + the tool catalog, then at each step returns a JSON
decision:
    {"thought": "...", "action": {"tool": "<name>", "args": {...}}}
    {"final": "<answer to the user>"}
    {"ask_user": "<clarifying question>"}      # ambiguous -> ASK, never guess
We dispatch the action, feed back a compact observation, and repeat — bounded to
max_tool_calls per turn (execution_policy: 8). This reuses the legacy Gemini client's
JSON output mode, so no separate function-calling client is needed.

run_intent() is the FALLBACK (HANDOVER §6): if the free loop is unstable, the UI calls
fixed tool sequences per intent. The LLM still does enrich/explain inside them.
"""

from __future__ import annotations

import json
import os
import re

import config
from ai_client import call_ai
from registry import TOOLS, call_tool, tool_catalog

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# askfix REBASE (owner decision): back to the S16 conversational frame — ONE explain per
# turn (≤8 phrases; Python appends the Summary and "explain the rest" runs deterministically),
# so a discovery turn needs only ~5 calls (ingest, extract, recall, wordnet, explain). 10 gives
# headroom without re-inviting long chains (the old 16 existed only to fund explain chaining).
# (Kept in sync with specs/config/execution_policy.yaml: max_tool_calls_per_turn — documentary,
# not code-enforced.)
MAX_TOOL_CALLS = 10


# --------------------------------------------------------------------------- #
# Static context (AGENTS.md + skills) — loaded once, injected into the system prompt
# --------------------------------------------------------------------------- #

def _read(path: str, limit: int = 4000) -> str:
    try:
        with open(os.path.join(_ROOT, path), "r", encoding="utf-8") as f:
            return f.read()[:limit]
    except Exception:
        return ""


def _system_prompt() -> str:
    agents = _read("AGENTS.md")
    skill_build = _read("skills/building-vocab-graph/SKILL.md", 2500)
    skill_expand = _read("skills/expanding-vocab-knowledge/SKILL.md", 2500)
    return (
        "You are VocabGraph-Agent, a careful PROCESS EXECUTOR (not a chatbot). "
        "You turn a learner's media/questions into a growing vocab graph + Anki cards "
        "by CHOOSING and calling tools. Obey the operating contract below.\n\n"
        f"=== AGENTS.md (supreme contract) ===\n{agents}\n\n"
        f"=== SKILL: building-vocab-graph ===\n{skill_build}\n\n"
        f"=== SKILL: expanding-vocab-knowledge ===\n{skill_expand}\n\n"
        "=== AVAILABLE TOOLS ===\n" + tool_catalog() + "\n\n"
        "=== RULES ===\n"
        "- ALWAYS call recall first before fetching anew.\n"
        "- When you must look up MANY terms (e.g. explaining several phrases), BATCH them: pass "
        "the whole comma-separated list in ONE `recall` call and ONE `wordnet_lookup` call — each "
        "tool splits the list and returns per-term results under `batch`. This SAVES your tool-call "
        "budget (do NOT spend one call per term). Use a single-term call "
        "only when you are looking up just one word. Explain the up-to-8 MOST useful phrases in "
        "ONE `explain` call (do NOT chain multiple explain calls — the system lists any remainder).\n"
        "- Deterministic-first: wordnet_lookup before enrich; never invent senses/edges.\n"
        "- Mining from media: after ingest_transcript, set extract_vocab's `transcript` "
        "argument to the `srt_path` string returned by ingest_transcript — do NOT paste the "
        "transcript text (it is truncated in observations, which gives too few words). AIM FOR "
        "8-20 learnable items — the tool "
        "already gathers until it has at least 8 unique (max_terms default 20). If it returns "
        "fewer than ~8, call it again with a larger max_terms or no focus before answering; never "
        "answer from only 2-3 terms. Prefer idioms / phrasal verbs / collocations / slang over "
        "common function words.\n"
        "- Tool ORDER: recall -> wordnet_lookup -> conceptnet_lookup (ONLY if WordNet is "
        "sparse/OOV) -> enrich. Call `explain` LAST, after the facts are gathered — never "
        "before the lookups it should be grounded on.\n"
        "- MANDATORY `explain`: whenever the learner asks what a word/phrase MEANS, or to "
        "define / explain / 'giải nghĩa' it (ANY meaning or 'what does X mean' request, in any "
        "language), you MUST answer by calling the `explain` tool (after recall + wordnet_lookup) "
        "— NEVER answer a meaning question from your own words without calling `explain`. This is "
        "what produces the grounded 3-layer answer (From your graph / From this video / "
        "Dictionary / Beyond this video). If the scratch shows REMAINING_UNEXPLAINED and the "
        "learner asks to continue, call `explain` on those terms (never hand-write the glosses).\n"
        "- A question about whether a word/phrase APPEARS in the video/script/transcript is a "
        "FACT question — answer it ONLY from tool observations (recall and any NOTE attached to "
        "it), never from memory. If a NOTE says the term DOES appear, answer YES and quote that "
        "exact line.\n"
        "- ONE `explain` call per turn, covering the up-to-8 MOST useful phrases (pick the best "
        "if more were found). Do NOT call `explain` a second time — after it returns, emit `final`. "
        "The system AUTOMATICALLY appends a summary listing the total found, which are already in "
        "the learner's collection, and which phrases were NOT explained yet (offering to continue) "
        "— so you do NOT need to write that list yourself, and must NEVER hand-write phrase "
        "explanations in place of the `explain` tool.\n"
        "- Before calling `explain`, pass the `recall()` hits as the `context` argument so the "
        "answer is grounded in what the learner already learned. When you have called `explain`, "
        "your `final` MUST contain that full explanation VERBATIM (do not shorten or drop it). "
        "Only AFTER giving the explanation may you add a short offer to save the word — never "
        "replace the explanation with just a follow-up question.\n"
        f"- Bounded: at most {MAX_TOOL_CALLS} tool calls this turn. If ambiguous, ASK.\n"
        "- Do NOT commit anything to the graph/deck — only draft (HITL review happens in the app).\n"
        "- You may call `stage_for_review` ONLY when the learner explicitly wants to keep a word — "
        "and ASK the learner first before staging (never stage unprompted). It writes to the review "
        "queue, not the graph. When the learner agrees to save several words, pass EVERY one of "
        "them in the `terms` list of a SINGLE stage_for_review call (e.g. terms=['find out','give "
        "up','go on', …]) — do not stage just one. For EACH term you stage, also pass the exact "
        "source line it appeared in via the `sentences` map (e.g. sentences={'find out':'How did "
        "you find out?'}) — this is the GROUNDING evidence. A term with no source sentence, or a "
        "sentence that does not actually contain the word, is REJECTED (returned in `ungrounded`); "
        "the sentence MUST be an EXACT line from the transcript — so call `ingest_transcript` on "
        "the media FIRST and copy the real line (never invent or paraphrase a sentence). If a term "
        "comes back `ungrounded`, the reason includes the real transcript line(s); use them to fix "
        "the sentence in ONE retry — do not make up a new sentence. VERIFY every sentence is a "
        "VERBATIM line from the ingested transcript BEFORE you finalize (the card shows this exact "
        "line and its audio is cut from that line's cue — a paraphrase desyncs the audio). After "
        "it returns, report to the learner EXACTLY the words it says were staged (the tool's "
        "`staged` list) — never claim to have saved words it did not stage.\n\n"
        "=== RESPONSE FORMAT (STRICT JSON, one object, no markdown) ===\n"
        'One of: {"thought":"..","action":{"tool":"<name>","args":{..}}}  |  '
        '{"final":"<answer>"}  |  {"ask_user":"<question>"}'
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_decision(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # F1: the model sometimes emits VALID JSON followed by trailing prose/data (json.loads then
    # dies with "Extra data" and kills the whole turn). raw_decode reads the FIRST complete JSON
    # value and ignores anything after it; strict=False still tolerates literal control chars
    # (raw newlines inside a long "explanation"/args string). One stray byte must not lose a turn.
    dec = json.JSONDecoder(strict=False)
    try:
        obj, _ = dec.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: leading prose before the object -> decode from the first "{" (still first-value
    # only, so a trailing tail after the object is harmless).
    brace = text.find("{")
    if brace >= 0:
        try:
            obj, _ = dec.raw_decode(text[brace:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # F3 (additive salvage, 0 API): the model sometimes breaks JSON with an UNESCAPED inner
    # quote inside its free-text answer ("final"/"ask_user"), so every decoder above fails and
    # the whole turn would be lost. Recover just that text (greedy to the last quote before the
    # closing brace/comma) instead of re-asking the model. Tool-action JSON almost never breaks,
    # so we only salvage the two text shapes; no match -> fall through to the strict load below.
    for _k in ("final", "ask_user"):
        _m = re.search(r'"' + _k + r'"\s*:\s*"(.*)"\s*[},]', text, re.DOTALL)
        if _m:
            return {_k: _m.group(1).strip()}
    # Last resort: strict load (raises -> caller turns it into a graceful "(agent error)").
    return json.loads(text, strict=False)


def _unwrap_final(final) -> str:
    """S17 (KNOWN LIMITATION ②): the agent sometimes double-wraps its answer — its `final`
    is itself a JSON envelope like `{"explanation": "..."}` (or a raw dict), so the learner
    saw literal JSON in chat. `explain` already gained a robust `_unwrap`; reuse it here at
    the agent's OUTPUT boundary. Deterministic-first: Python peels the envelope in-process —
    NO extra AI round-trip (an envelope is a formatting slip, not a content error). Normal
    markdown prose is returned unchanged (it never parses as an envelope dict)."""
    if isinstance(final, dict):
        for k in ("final", "explanation", "text", "answer"):
            if isinstance(final.get(k), str):
                final = final[k]
                break
        else:
            return str(final)
    try:
        from explain import _unwrap
        return _unwrap(str(final))
    except Exception:
        return str(final)


def _observe(result) -> str:
    """Compact, token-cheap observation string fed back to the model."""
    try:
        if isinstance(result, dict):
            # S17-5.1a: a comma-batch input was split by the tool (recall/wordnet_lookup)
            # into per-term results under "batch" — observe EACH term so the model gets
            # per-term found/provenance instead of one compacted dict, plus the tool's note.
            batch = result.get("batch")
            if isinstance(batch, dict) and batch:
                per_term = "; ".join(f"[{t}] {_observe(r)}" for t, r in batch.items())
                note = str(result.get("note", ""))
                return (f"BATCH of {len(batch)} terms ({note}): " + per_term)[:4000]
            # S16 T-A3: for a recall hit, the generic dict compaction below drops source_map /
            # occurrences (lists become "[N items]"), so the model can't answer provenance
            # questions ("who defined X? which film?"). Attach a compact provenance summary of
            # the main node (additive; the rest of the observation stays token-cheap).
            main = result.get("as_main_node")
            prov = ""
            if isinstance(main, dict) and main:
                smap = main.get("source_map") or {}
                srcs, seen_src = [], set()
                for occ in (main.get("occurrences") or []):
                    s = occ.get("source", "") if isinstance(occ, dict) else ""
                    if s and s not in seen_src:
                        seen_src.add(s)
                        srcs.append(s)
                    if len(srcs) >= 4:
                        break
                prov_obj = {
                    "term": main.get("term", ""),
                    "definition_source": smap.get("definition", ""),
                    "sense_id": main.get("sense_id") or "",
                    "sources": srcs,
                    "n_occurrences": len(main.get("occurrences") or []),
                }
                prov = (" provenance=" + json.dumps(prov_obj, ensure_ascii=False)
                        + " ALREADY A LEARNED WORD — tell the learner they know this word and "
                          "where they met it; you MAY summarize/refresh what they already know "
                          "(definition, films) before adding new information. Also point them to "
                          "where they can REVIEW it in the app: rewatch its highlighted (yellow) "
                          "lines via the exported subtitle file (.ass) of that film, see the "
                          "word's node in the Graph view, and practice it with the exported Anki "
                          "deck — then invite a follow-up question if they want more.")
            elif "as_related" in result:
                # S16+ (T-A3 TH2): the word is NOT a learned node, but the graph has SEEN it —
                # as another node's edge target, inside a stored sentence, or in a collocation.
                # Surface those associative hits so the agent tells the learner where it
                # appeared instead of treating it as brand new.
                # as_related holds node KEYS ("tusk#tusk.n.01"); tolerate dicts too.
                related = []
                for n in (result.get("as_related") or []):
                    t = (n.get("term", "") if isinstance(n, dict)
                         else str(n).split("#", 1)[0])
                    if t:
                        related.append(t)
                    if len(related) >= 4:
                        break
                n_sent = len(result.get("in_sentences") or [])
                n_coll = len(result.get("in_collocations") or [])
                if related or n_sent or n_coll:
                    seen_obj = {"related_to": related,
                                "in_sentences": n_sent, "in_collocations": n_coll}
                    prov = (" seen_before=" + json.dumps(seen_obj, ensure_ascii=False)
                            + " SEEN BEFORE as a RELATED word (not yet learned) — tell the "
                              "learner where it appeared (e.g. related to those words) before "
                              "explaining.")
            # S18 (owner request): a word can sit in the REVIEW QUEUE without being in the
            # graph yet. Tell the model so it says "staged, awaiting your approval in the
            # Review tab" instead of the false "you haven't learned this".
            if not result.get("found") and result.get("in_review_queue"):
                flagged = any(e.get("flagged_ungrounded") for e in result["in_review_queue"]
                              if isinstance(e, dict))
                prov += (" IN REVIEW QUEUE — this word is ALREADY STAGED and awaiting the "
                         "learner's approval in the Review tab (NOT yet committed to the "
                         "graph). Tell the learner it is waiting for their review"
                         + (" and is flagged '⚠ ungrounded' (needs a real source sentence "
                            "before it can commit)" if flagged else "")
                         + " — do NOT say they haven't learned/saved it.")
            small = {}
            for k, v in result.items():
                if isinstance(v, list):
                    small[k] = f"[{len(v)} items]"
                elif isinstance(v, (dict,)):
                    small[k] = f"{{{', '.join(list(v.keys())[:5])}}}"
                else:
                    s = str(v)
                    small[k] = s if len(s) <= 160 else s[:160] + "…"
            return json.dumps(small, ensure_ascii=False) + prov
        if isinstance(result, list):
            # If these are candidate/vocab items, surface their TERMS (not just a count) so
            # the agent can reason about a freshly-ingested movie's words and explain them.
            if result and isinstance(result[0], dict) and "term" in result[0]:
                terms = ", ".join(str(x.get("term", "")) for x in result[:15])
                return f"[{len(result)} items] terms: {terms}"
            return f"[{len(result)} items] " + json.dumps(result[:3], ensure_ascii=False)[:400]
        s = str(result)
        return s if len(s) <= 400 else s[:400] + "…"
    except Exception:
        return "<result>"


def _norm_loc(s: str) -> str:
    """Same [a-z0-9]+ normalization as extract_vocab._grounded — so a term confirmed
    grounded in the transcript is locatable here with identical matching rules."""
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


_REVIEW_PTR = ("review them in the **Graph view**, the **yellow-highlighted lines** of the "
               "exported subtitle (.ass), or your **Anki deck**")


# S19 BUG-2: edge types we surface as a genuine cross-graph link. `antonym` is DROPPED —
# WordNet matches by LEMMA not SENSE, so an antonym edge on a same-spelled DIFFERENT sense
# ("turn in"=go to bed ↔ "turn out"=get out of bed) is a FALSE link for the sense the learner
# actually studied ("turn out"=result). synonym / is_a / hyponym / part_of / category stay —
# a same-lemma match there is far likelier to be genuinely related.
_LINK_KEEP_TYPES = {"synonym", "is_a", "hyponym", "part_of", "category"}


def _antonym_on_primary_sense(word: str, opposite: str) -> bool:
    """S19-S4: True iff `opposite` is a WordNet antonym on the PRIMARY (first) synset of `word`.

    This rescues sense-CORE antonyms — e.g. sane↔insane, where "insane" is the antonym of
    sane's very first synset (sane.a.01) — while still dropping the BUG-2 false link: "turn
    out"'s first synset is prove.v.01 ("result", the sense the learner studied) which has NO
    antonym; its turn_in antonym lives only on a peripheral "get out of bed" sense. Deterministic,
    0 AI. Any lookup failure returns False (fall back to the conservative drop)."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False
    w = (word or "").strip().lower().replace(" ", "_")
    opp = (opposite or "").strip().lower().replace(" ", "_")
    if not w or not opp:
        return False
    syns = wn.synsets(w)
    if not syns:
        return False
    for lm in syns[0].lemmas():                 # primary synset only
        if lm.name().lower() == w:
            if any(a.name().lower() == opp for a in lm.antonyms()):
                return True
    return False


def _resolve_cross_links(related_links: dict | None) -> list[str]:
    """S19 BUG-2 — turn {new_term -> {existing_node_key,...}} into learner-facing lines that
    NAME the relation type and the existing node's meaning, instead of a bare `X ↔ Y`. Loads
    the graph once (0 AI, deterministic) to read the real edge type + definition, and drops
    antonym / non-relational edges (the sense-mismatch noise the owner hit). Returns [] if the
    graph is unavailable or nothing survives filtering."""
    if not related_links:
        return []
    try:
        from _common import load_graph, GRAPH_PATH
        nodes = getattr(load_graph(GRAPH_PATH), "nodes", {}) or {}
    except Exception:
        return []
    lines = []
    for ql, rkeys in related_links.items():
        for rkey in sorted(rkeys):
            node = nodes.get(rkey)
            if node is None:                       # tolerate a bare-lemma key
                rk_term = str(rkey).split("#")[0].strip().lower()
                node = next((n for n in nodes.values()
                             if (getattr(n, "term", "") or "").strip().lower() == rk_term), None)
            if node is None:
                continue
            rt = (getattr(node, "term", "") or str(rkey).split("#")[0]).strip()
            if rt.lower() == ql:
                continue
            etype = ""                             # the edge on THIS node pointing back at ql
            for e in (getattr(node, "edges", []) or []):
                if (getattr(e, "target", "") or "").strip().lower() == ql:
                    etype = (getattr(e, "type", "") or "").strip().lower()
                    break
            if etype and etype not in _LINK_KEEP_TYPES:
                # S19-S4: a sense-CORE antonym (opposite is the antonym of ql's PRIMARY synset,
                # e.g. sane↔insane) is trustworthy and worth surfacing; a peripheral-sense
                # antonym (turn out↔turn in) stays dropped to avoid the BUG-2 false link.
                if not (etype == "antonym" and _antonym_on_primary_sense(ql, rt)):
                    continue                       # drop antonym-noise & anything not related
            gloss = (getattr(node, "definition", "") or "").strip()
            if len(gloss) > 61:
                gloss = gloss[:60] + "…"
            rel = etype or "related"
            lines.append(f"- **{ql}** —{rel}→ **{rt}**" + (f" (meaning: {gloss})" if gloss else ""))
            if len(lines) >= 6:
                return lines
    return lines


def _summary_block(extracted_cands: list, explained_terms: list, known_terms: dict,
                   related_links: dict | None = None) -> str:
    """askfix (owner #1): the deterministic end-of-turn summary (0 AI, so counts are always
    accurate). On a DISCOVERY turn (extract_vocab ran -> `extracted_cands`) it reports the total
    found, which are ALREADY saved (`known_terms`: lower-term -> where), and which phrases were
    NOT explained yet (offering to continue) — so a capped answer is never a dead end. On a plain
    meaning question about an already-saved word it shows just the review pointer. Returns '' when
    there is nothing worth appending."""
    found = list(dict.fromkeys(c.get("term", "") for c in (extracted_cands or []) if c.get("term")))
    exp_norm = {_norm_loc(t) for t in (explained_terms or [])}
    known_terms = known_terms or {}
    if found:
        # askfix S19 (#1): MUTUALLY-EXCLUSIVE buckets so the counts reconcile (old bug: a term
        # explained THIS turn that was also already in the graph was counted BOTH as "explained"
        # and "already learned", so "Found 28 / Explained 9 / Learned 9 / Not 14" didn't add up).
        # A term is now in exactly ONE bucket: explained-this-turn > already-known > remaining.
        explained = [t for t in found if _norm_loc(t) in exp_norm]
        known = [t for t in found if t.lower() in known_terms and _norm_loc(t) not in exp_norm]
        remaining = [t for t in found if _norm_loc(t) not in exp_norm and t.lower() not in known_terms]
        n_explained = len(explained)   # explained + len(known) + len(remaining) == len(found)
        L = ["\n\n---\n\n### 📋 Summary",
             f"- Found **{len(found)}** useful phrase(s) in this video."]
        if n_explained:
            L.append(f"- Explained **{n_explained}** above.")
        if known:
            # owner: "collection" is ambiguous — say WHERE each word actually is. Committed
            # graph words are studyable (graph/.ass/Anki); queue words still await approval.
            in_graph = [t for t in known if known_terms.get(t.lower()) != "review queue"]
            in_queue = [t for t in known if known_terms.get(t.lower()) == "review queue"]
            if in_graph:
                L.append(f"- Already LEARNED (in your graph): {', '.join(in_graph)} — {_REVIEW_PTR}.")
            if in_queue:
                L.append(f"- Waiting in the REVIEW QUEUE (not yet approved): {', '.join(in_queue)} "
                         f"— approve them in the Review tab to add them to your graph/Anki.")
        if remaining:
            L.append(f"- Not explained yet ({len(remaining)}): {', '.join(remaining)} — "
                     f"say \"explain the rest\" to continue, or \"save them to my queue\" to keep them.")
        # S19 (owner): surface CROSS-graph links — a word in THIS video that connects (synonym /
        # is_a / …) to a word already in the learner's graph (possibly from another film). Flag
        # only (0 AI, no fabrication); depth is on demand via a normal follow-up question. One
        # link per line. Skips words already reported as "Already LEARNED" (they're exact repeats).
        links = _resolve_cross_links(related_links)
        if links:
            L.append("\n**🔗 Connects to your graph:**")
            L.extend(links)
            # follow-up hint uses the first link's two terms (now that the relation is explicit)
            _m = re.search(r"\*\*(.+?)\*\* —.+?→ \*\*(.+?)\*\*", links[0])
            if _m:
                L.append(f"_Ask \"how does {_m.group(1)} relate to {_m.group(2)}?\" for details._")
        return "\n".join(L)
    if known_terms:                       # plain question about an already-saved word
        g = sorted(t for t, w in known_terms.items() if w != "review queue")
        q = sorted(t for t, w in known_terms.items() if w == "review queue")
        parts = []
        if g:
            parts.append(f"{', '.join(g)} — already in your graph; {_REVIEW_PTR}")
        if q:
            parts.append(f"{', '.join(q)} — in the review queue awaiting your approval (Review tab)")
        return "\n\n---\n\n💡 **Review it:** " + "; ".join(parts) + "."
    # S19-S4: a single-word question ("explain 'sane' in this scene") runs NO discovery and the
    # word may not be a node yet, so both branches above are empty — but recall can still have
    # found a real cross-graph link (sane → insane antonym in another film). Surface it here so
    # the hero cross-film moment isn't lost just because the turn wasn't a batch discovery.
    links = _resolve_cross_links(related_links)
    if links:
        L = ["\n\n---\n\n### 🔗 Connects to your graph:"]
        L.extend(links)
        _m = re.search(r"\*\*(.+?)\*\* —.+?→ \*\*(.+?)\*\*", links[0])
        if _m:
            L.append(f"_Ask \"how does {_m.group(1)} relate to {_m.group(2)}?\" for details._")
        return "\n".join(L)
    return ""


def _next_step_hint(kind: str) -> str:
    """askfix S19 (A3): a single deterministic next-step line (0 AI, so it never fabricates and
    never dead-ends the conversation). `kind`: 'explained_all' (offer to save) | 'staged' (point
    to the Review tab). The discovery case is already handled inside `_summary_block`."""
    if kind == "staged":
        return ("\n\n👉 Next: open the **Review** tab to approve them — then **Commit** to add "
                "them to your Anki deck + graph.")
    if kind == "explained_all":
        return ("\n\n👉 Next: say \"save them to my queue\" to keep any of these, or ask about "
                "another word.")
    return ""


_EXPLAIN_FIELD_LABELS = (
    "Meaning", "Definition", "Examples", "Example", "Synonyms", "Antonyms", "Register",
    "Pronunciation", "Collocations", "Common Collocations", "Common Mistakes", "Common Mistake",
    "Mnemonic", "Tags", "Note", "Usage", "From this video", "From your graph")
# a field label glued onto the tail of the previous line: preceded by a NON-space char + space(s),
# with an optional leading ** (bold). Word-boundary + the exact Capitalized label set keeps prose
# ("...the register of...") from matching. We only split when it is NOT already at line start.
_EXPLAIN_FIELD_RE = re.compile(
    r"(?<=\S)\s+(?=\*{0,2}(?:" + "|".join(re.escape(l) for l in _EXPLAIN_FIELD_LABELS) + r")\*{0,2}\s*:)")


def _format_explain_fields(text: str) -> str:
    """askfix S19 (#2): the AI sometimes runs several fields onto one line ("...support oneself.
    Register: Neutral. Pronunciation: /.../") OR joins them with a SINGLE newline, which Markdown
    renders as a space (still glued). Put each known field label on its own line with a BLANK line
    before it (\\n\\n) so it actually renders as a separate line. DETERMINISTIC display fix (0 AI),
    applied to learner-facing text only (Anki/graph fields come from node data, never from this
    rendered text). Any run of whitespace before a glued label collapses to one blank line, so it
    is idempotent on already-well-formatted text. Content words are never touched."""
    if not text:
        return text
    return _EXPLAIN_FIELD_RE.sub("\n\n", text)


def _renumber_terms(text: str) -> str:
    """S18 askfix (owner #5): each chained `explain` batch numbers its own terms 1..5, so the
    joined learner answer reads 1-5, 1-5, 1-5. Renumber the TERM HEADINGS sequentially across
    the whole answer (1..N). Deterministic; only the two heading shapes explain actually emits
    are touched — '### 1. term' (markdown heading) and '1. **term**' (bold list item) at line
    start. Bullet examples ('*  ...') and prose match neither shape, so content is untouched."""
    n = 0
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(#{2,4}\s*)\d+([.)]\s+)", line)
        if m:
            n += 1
            line = f"{m.group(1)}{n}{m.group(2)}{line[m.end():]}"
        else:
            m2 = re.match(r"^\d+([.)]\s+\*\*)", line)
            if m2:
                n += 1
                line = f"{n}{m2.group(1)}{line[m2.end():]}"
        out.append(line)
    return "\n".join(out)


def _dedup_final(joined: str, answer) -> str:
    """S18 HEART §1e: return the model's trailing `final` text, or "" when it merely REPEATS
    the already-concatenated explanations (so it isn't printed twice). Recap detection is
    conservative — a substantive sentence (≥4 content words) is 'seen' only when it appears
    VERBATIM (normalized) in `joined`; if ≥70% of the trailing text's substantive sentences are
    seen, it's a recap -> drop. A short/novel follow-up ('want me to save these?') survives."""
    extra = (answer or "").strip()
    if not extra or extra in joined:
        return ""

    def _sents(t):
        return [k for k in (_norm_loc(p) for p in re.split(r"[.\n?!…]+", t))
                if len(k.split()) >= 4]
    seen = set(_sents(joined))
    es = _sents(extra)
    if es and sum(1 for s in es if s in seen) / len(es) >= 0.7:
        return ""
    # PARAPHRASED per-term recap (live Q1: the final re-listed all 14 explained terms as
    # '**term**: one-line gloss' under '### Batch N' headings — not verbatim, so the sentence
    # check missed it). Drop each recap-SHAPED line (**bold term** at line start followed by
    # ':'/'—') whose term was already explained; then drop headings left with no content.
    jn = _norm_loc(joined)
    lines = extra.split("\n")
    pruned = set()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(?:[-*\d.)]+\s*)*\*\*(.+?)\*\*\s*[:—–-]", line)
        if m and (t := _norm_loc(m.group(1))) and f" {t} " in f" {jn} ":
            pruned.add(i)
    out = []
    for i, line in enumerate(lines):
        if i in pruned:
            continue
        if re.match(r"^\s*#{1,6}\s", line):    # heading whose own items were all recap-pruned
            j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
            if j is None or j in pruned:
                continue
        out.append(line)
    extra = "\n".join(out).strip()
    return extra


def _collect_repeats(node: dict, segments: list, cap: int = 10) -> None:
    """Append an Occurrence for EVERY other line where this node's word recurs, keeping each
    line's ORIGINAL surface form so no source data is lost (extract dedups a term to ONE
    candidate, but a word recurs — often in different inflections).

    Single-word nodes match by LEMMA (so "reduce"/"reduced"/"reduces" all count as the SAME
    node, each occurrence recording its real surface); multi-word nodes match the primary
    surface phrase verbatim (phrasal inflection across separated tokens is too noisy to do
    deterministically). Mutates node['occurrences'] in place; never crashes (no-crash §4).
    """
    occ = node.get("occurrences") or []
    if not occ:
        return
    term = (node.get("term") or "").strip().lower()
    primary_surface = occ[0].get("surface", "") or term
    if not term:
        return
    try:
        from extract_vocab import lemmatize_term
    except Exception:
        lemmatize_term = None
    multiword = " " in term
    seen_sents = {_norm_loc(o.get("sentence", "")) for o in occ}
    src, added = occ[0].get("source", ""), occ[0].get("added_at", "")
    for seg in segments or []:
        if len(node["occurrences"]) >= cap:
            break
        text = seg.get("text", "")
        sn = _norm_loc(text)
        if not sn or sn in seen_sents:
            continue
        matched = ""
        if multiword:
            if _norm_loc(primary_surface) in sn:
                matched = primary_surface
        else:
            for tok in re.findall(r"[A-Za-z']+", text):
                if tok.lower() == term or (lemmatize_term and lemmatize_term(tok) == term):
                    matched = tok               # the ORIGINAL form as it appears in THIS line
                    break
        if not matched:
            continue
        seen_sents.add(sn)
        node["occurrences"].append({
            "source": src, "sentence": text, "surface": matched,
            "lemmas": [], "media": {}, "added_at": added,
            "start": seg.get("start", ""), "end": seg.get("end", ""),
        })


def _hms(v) -> str:
    """Format a value as "HH:MM:SS" for occurrence PROVENANCE display. A float (seconds) is
    rounded to the second (display-only — the clip keeps the raw float); a string passes
    through unchanged; anything else -> ""."""
    if isinstance(v, bool):
        return ""
    if isinstance(v, (int, float)):
        s = int(max(0.0, float(v)))
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return v if isinstance(v, str) else ""


def _sec_of(v):
    """A timing value (float seconds, or 'HH:MM:SS[.mmm]', or '') -> float seconds or None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", v.strip()):
        a, b, c = v.strip().split(":")
        return int(a) * 3600 + int(b) * 60 + float(c)
    return None


_NEIGHBOR_SNAP = 1.0   # only expand when the nearest cue boundary is within this many seconds


def _expand_to_neighbor(segments: list, start, end) -> tuple:
    """S18/S19: widen a located (start,end) to the PREVIOUS block's start and the NEXT block's
    end, so a short cue isn't clipped and carries ~1 subtitle of context each side ('dài còn
    hơn thiếu'). Returns FLOAT seconds; degrades to the input when no cue matches.

    S19 (owner "mất âm cuối dòng"): each edge now snaps to its NEAREST cue boundary instead of
    requiring an exact <0.05s equality. The old exact match BAILED whenever the clip's start
    drifted a few ms (agent re-location / rounding), leaving the bare cue + only make_anki's
    0.2s pad -> the last word got clipped. Ported from the reference project's
    expand_sub_context (nearest-boundary, side-aware). A per-edge sanity threshold
    (_NEIGHBOR_SNAP) still refuses to snap a wildly-off timestamp onto an unrelated cue."""
    s0, e0 = _sec_of(start), _sec_of(end)
    if s0 is None or e0 is None:
        return start, end
    segs = segments or []
    if not segs:
        return s0, e0

    def _nearest(target, sec_key, str_key):
        best_i, best_d = None, None
        for i, seg in enumerate(segs):
            raw = seg.get(sec_key)
            v = _sec_of(raw if raw not in ("", None) else seg.get(str_key))
            if v is None:
                continue
            d = abs(v - target)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    si, sd = _nearest(s0, "start_sec", "start")     # cue whose START is closest to our start
    ei, ed = _nearest(e0, "end_sec", "end")         # cue whose END is closest to our end
    ns = ne = None
    if si is not None and sd is not None and sd <= _NEIGHBOR_SNAP and si > 0:
        ns = _sec_of(segs[si - 1].get("start_sec") if segs[si - 1].get("start_sec") not in ("", None)
                     else segs[si - 1].get("start"))
    if ei is not None and ed is not None and ed <= _NEIGHBOR_SNAP and ei + 1 < len(segs):
        ne = _sec_of(segs[ei + 1].get("end_sec") if segs[ei + 1].get("end_sec") not in ("", None)
                     else segs[ei + 1].get("end"))
    return (ns if ns is not None else s0), (ne if ne is not None else e0)


def _seg_time(seg: dict) -> tuple:
    """The (start, end) of a segment, preferring FLOAT seconds (start_sec/end_sec) so the
    clip keeps millisecond precision. Falls back to the "HH:MM:SS" strings when the segment
    is untimed (start_sec=="") or lacks the float keys (legacy callers / tests)."""
    s, e = seg.get("start_sec"), seg.get("end_sec")
    if isinstance(s, (int, float)) and not isinstance(s, bool) \
            and isinstance(e, (int, float)) and not isinstance(e, bool):
        return s, e
    return seg.get("start", ""), seg.get("end", "")


def _locate_timestamp(segments: list, anchor: str, sentence: str) -> tuple:
    """Find the source (start, end) of the line a mined term came from.

    Returns FLOAT seconds when the matched segment is timed (start_sec/end_sec present),
    else the "HH:MM:SS" strings, else ("","") — see _seg_time.

    Three deterministic tiers (no AI), most precise first. S18 P0-1b — the displayed
    SENTENCE now wins over a verbatim surface match: a repeated phrase ("make it work")
    appears in MANY cues, so anchoring on the surface pulled the FIRST such cue, not the
    cue of the card's own sentence -> audio said a different line than the card showed.
    The cited sentence identifies exactly ONE cue, so it is authoritative:
      1) the cited SENTENCE overlaps a segment -> that exact line (authoritative);
      2) verbatim SURFACE appears in a segment -> that line (when no sentence given);
      3) (S12 T2, tightened S14 T9) the LONGEST CONTENT token of the surface appears in a
         segment -> first such line. Stopwords are excluded — matching "be"/"a"/"in" put a
         WRONG timestamp on the first stopword-bearing line, worse than no timestamp. A
         surface made only of stopwords skips tier 3 entirely.
    Returns ("","") when nothing content-bearing matches (a wrong time is worse than none).
    """
    na, ns = _norm_loc(anchor), _norm_loc(sentence)
    from _common import stopwords_set
    stop = stopwords_set()
    content = [t for t in na.split() if t not in stop]
    anchor_tok = max(content, key=len) if content else ""   # longest content token
    surface_hit = None
    token_fallback = None
    for seg in segments or []:
        sn = _norm_loc(seg.get("text", ""))
        if not sn:
            continue
        if ns and (ns in sn or sn in ns):         # displayed sentence -> its exact cue
            return _seg_time(seg)
        if surface_hit is None and na and na in sn:   # surface verbatim (no sentence)
            surface_hit = _seg_time(seg)
        if token_fallback is None and anchor_tok and anchor_tok in set(sn.split()):
            token_fallback = _seg_time(seg)
    return surface_hit or token_fallback or ("", "")


def _materials_for(query: str, source: str, extracted: list | None = None,
                   with_scene: bool = False) -> dict:
    """S18 #6 / HEART §1a: build the explain 'From this video' context — for each term that will
    actually be explained this turn, its VERBATIM source cue. Python owns the cue two ways, in
    order: (1) REUSE the real source line extract_vocab already captured for a discovered
    candidate (its `sentence` is transcript-grounded — don't re-match it); (2) else the SHARED
    ground_line() helper over the cached transcript. Deterministic (0 AI), keyed by the ①
    transcript cache (source basename). Returns {} when nothing grounds so explain claims nothing
    from the video (a missing quote is better than a fabricated one)."""
    if not source:
        return {}
    try:
        import os
        import re as _re
        from _common import load_cached_transcript, ground_line
        transcript = load_cached_transcript(source)
        if not transcript:
            return {}
        # A cached transcript can be one giant paragraph (no per-cue newlines), so split on
        # sentence ends TOO — a "line" fed to explain must be ONE quotable sentence, not a blob
        # the model cherry-picks from. (song lyrics are ♪-delimited, so ♪ is a separator too)
        lines = [ln.strip() for ln in _re.split(r"(?<=[.?!…])\s+|[\r\n]+|♪", transcript)
                 if ln.strip()]

        def _norm(s):
            return " ".join(_re.findall(r"[a-z0-9']+", str(s).lower()))
        # (1) map every extract_vocab candidate's term/surface -> its already-grounded cue, so a
        # phrase DISCOVERED by the miner (not present in `query`) still carries its real line.
        cand_line = {}
        for c in (extracted or []):
            if not isinstance(c, dict):
                continue
            ln = str(c.get("sentence", "") or "").strip()
            if not ln:
                continue
            for k in (c.get("term"), c.get("surface")):
                kn = _norm(k)
                if kn and kn not in cand_line:
                    cand_line[kn] = ln
        hits, seen = [], set()
        for term in [t.strip() for t in str(query or "").split(",") if t.strip()][:10]:
            low = term.lower()
            if low in seen:
                continue
            # (2) reuse the miner's verbatim cue, else ground it via the shared helper.
            line = cand_line.get(_norm(term)) or ground_line(term, lines)
            if line:
                seen.add(low)
                hit = {"term": term, "line": line}
                # FEATURE-3 (lazy): when the learner asks for the meaning IN THE SCENE, attach a
                # BOUNDED window of neighbouring sentences (±_SCENE_WINDOW) so explain reads a
                # whole exchange, not just one line — but NOT the full script (a small model loses
                # focus / bloats quota on thousands of lines, and the quote itself is Python-owned).
                if with_scene:
                    idx = next((i for i, l in enumerate(lines)
                                if l == line or _norm(l) == _norm(line)), -1)
                    if idx >= 0:
                        w = _SCENE_WINDOW
                        before = [x for x in lines[max(0, idx - w):idx] if x]
                        after = [x for x in lines[idx + 1:idx + 1 + w] if x]
                        if before:
                            hit["scene_before"] = before
                        if after:
                            hit["scene_after"] = after
                hits.append(hit)
        if not hits:
            return {}
        return {"source": os.path.basename(source), "hits": hits}
    except Exception:
        return {}


_VIDEO_QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')

# FEATURE-3 (lazy): the learner explicitly wants the meaning AS USED IN THE SCENE (not just the
# dictionary sense). Matches EN + VI phrasings; kept narrow so a normal "explain X" turn is
# unaffected (in-scene context is opt-in — it makes the answer longer).
_IN_SCENE_RE = re.compile(
    r"\bin (?:the |this )?(?:film|movie|scene|clip)\b|\bin (?:the |this )?context\b"
    r"|\bin the video\b|trong (?:phim|c[aả]nh|ng[uữ] c[aả]nh)",
    re.IGNORECASE)

# FEATURE-3: how many sentences on EACH side of the real line to hand explain as scene context.
# ±5 (~11 lines) covers a full back-and-forth exchange without dumping the whole script.
_SCENE_WINDOW = 5

# askfix REBASE: explain's graceful-degrade outputs (tools/explain.py returns these strings
# instead of raising when the AI call fails — e.g. 429 quota exhausted past all backoff
# retries). They must NOT be treated as a delivered explanation: ending the turn on one would
# mark the terms explained and silently drop them from REMAINING_UNEXPLAINED.
_EXPLAIN_FAIL_RE = re.compile(r"^\(\s*(Could not generate an explanation|No explanation produced)")


def _explain_ok(res) -> bool:
    """True iff an `explain` result is real explanation text (not empty / not the tool's
    error sentinel)."""
    return isinstance(res, str) and bool(res.strip()) and not _EXPLAIN_FAIL_RE.match(res.strip())


def _recall_miss_note(result: dict, args: dict, sources) -> str:
    """askfix D3 (FIX B — anti-lie, deterministic, 0 AI): recall searches the learner's GRAPH,
    not the transcript — so `found=False` made the model tell the learner a phrase is "not in
    the script" even when it IS (live: 'crying out loud' @ 00:08:05). For each missed lemma,
    Python checks the cached transcript(s) itself via the shared ground_line(); a hit appends
    an explicit NOTE with the verbatim cue so the model cannot answer NO, and knows to pass
    this exact sentence to stage_for_review if the learner asked to save. Returns "" when
    nothing missed / no cached transcript / no match (observation unchanged)."""
    try:
        from _common import load_cached_transcript, ground_line
    except Exception:
        return ""
    missed, found = [], []
    batch = result.get("batch")
    if isinstance(batch, dict) and batch:
        for t, r in batch.items():
            if isinstance(r, dict) and str(t).strip():
                (found if r.get("found") else missed).append(str(t).strip())
    else:
        q = str(args.get("lemma") or args.get("term") or "")
        terms = [t.strip() for t in q.split(",") if t.strip()]
        (found if result.get("found") else missed).extend(terms)
    if not missed and not found:
        return ""
    lines = []
    for s in sources or []:
        t = load_cached_transcript(s)
        if t:
            # same one-quotable-sentence split as _materials_for
            lines += [ln.strip() for ln in re.split(r"(?<=[.?!…])\s+|[\r\n]+|♪", t) if ln.strip()]
    if not lines:
        return ""
    notes = []
    for L in missed[:8]:
        cue = ground_line(L, lines)
        if cue:
            notes.append(
                f' NOTE: recall searches the learner\'s GRAPH only. \'{L}\' DOES appear in the '
                f'ingested transcript: "{cue}". If the learner asked whether it is in the '
                f'script, answer YES with this exact line; if they asked to save it, call '
                f'stage_for_review with this sentence.')
    # live T3 regression: a FOUND word made the model INVENT script lines (it had the graph
    # hit but never the transcript cue). Hand it the real line so any script citation is
    # verbatim — never from memory.
    for L in found[:8]:
        cue = ground_line(L, lines)
        if cue:
            notes.append(
                f' NOTE: in THIS ingested transcript \'{L}\' appears in the exact line: '
                f'"{cue}". When citing the script, quote ONLY this verbatim line — never '
                f'invent or paraphrase script lines.')
    return "".join(notes)


_PROVENANCE_HEADINGS = ("from this video", "from your graph")
_NONPROV_HEADINGS = ("dictionary", "beyond this video", "beyond your", "general knowledge")
# Lines that quote for EXPLANATION, not provenance ("**Meaning:** ...", "*Example 1:* ...") —
# live Q2 showed the guard clobbering a Meaning line's paraphrase-quotes; only lines CLAIMING a
# real source line are policed.
_EXPLAIN_LABEL_RE = re.compile(
    r"^[\s*_\-#>\d.)]*(meaning|usage|use it|examples?|tip|note|structure|synonyms?|"
    r"antonyms?|pronunciation|register|common mistakes?|collocations?|grammar)\b", re.IGNORECASE)


def _verify_video_quotes(text: str, sources, cues: dict, real_lines) -> str:
    """S18 HEART §1b (askfix-generalized) — Python is the LAST word on every quote the model
    attributes to a REAL source: the 'From this video' AND 'From your graph' provenance layers.
    The model fabricates source lines (LEAD verified ~9/13 on the video path; real test showed it
    also invents graph lines on recall-only follow-up turns, e.g. 'I should get a say in this,
    too' which is not in the transcript). Deterministic hard backstop (0 AI): inside a provenance
    section, any quoted sentence that is NOT a real line is REPLACED — by the term's real cue when
    one is named on that line, else exactly 'Not found in this video.'.

    Ground truth (works even when nothing was ingested THIS turn):
      • `sources` — cached transcripts (by basename) of any film touched this turn;
      • `real_lines` — real occurrence sentences surfaced by recall/extract this turn;
      • `cues` — {normalized term -> its real line} for the REPLACEMENT text.
    Text outside the provenance sections (Dictionary / Beyond / examples) is untouched.
    Degrades to `text` on any error / when there is no ground truth to check against."""
    try:
        from _common import load_cached_transcript, _ground_norm
    except Exception:
        return text
    if not text:
        return text
    blobs = []
    for s in (sources or []):
        if s:
            t = load_cached_transcript(s)
            if t:
                blobs.append(_ground_norm(t))
    norm_tr = "  ".join(blobs)
    real_norms = {n for n in (_ground_norm(x) for x in (real_lines or [])) if n}
    grounded = {}
    for k, v in (cues or {}).items():
        nk, ln = _ground_norm(k), str(v or "").strip()
        if nk and ln:
            grounded[nk] = ln
    if not norm_tr and not real_norms and not grounded:
        return text                                  # no ground truth -> leave the answer alone
    NOT_FOUND = "Not found in this video."

    def _is_real(nq: str) -> bool:
        if norm_tr and nq in norm_tr:
            return True
        return any(nq in rn or rn in nq for rn in real_norms)

    def _fix_line(line: str) -> str:
        norm_line = _ground_norm(line)

        def _repl(m):
            q = m.group(1) if m.group(1) is not None else m.group(2)
            nq = _ground_norm(q)
            # Only police QUOTES that claim to be a source LINE (a sentence), not a short term
            # label ("fire away"): ≥4 words or ends in sentence punctuation.
            if not nq or not (len(nq.split()) >= 4 or q.rstrip().endswith((".", "!", "?", "…"))):
                return m.group(0)
            if _is_real(nq):                          # a real line -> keep verbatim
                return m.group(0)
            for t, cue in grounded.items():           # fabricated -> real cue of a term named here
                if t and t in norm_line:
                    return '"' + cue + '"'
            return NOT_FOUND                           # no cue -> honest "not found"
        return _VIDEO_QUOTE_RE.sub(_repl, line)

    # LINE-LOCAL policing (askfix fix): only a line that ITSELF claims a source ("From this
    # video"/"From your graph") is policed, plus the cue bullets DIRECTLY under such a heading
    # (block format: "From this video:\n- term: <quote>"). A run of cue bullets ends at the first
    # non-bullet line. This never bleeds into Meaning/Dictionary/Examples/Note lines — which was
    # corrupting them into "an idiom for Not found in this video." under the per-term layout.
    out, police_bullets = [], False
    for line in text.split("\n"):
        low = line.lower()
        is_bullet = bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s", line))
        if any(h in low for h in _PROVENANCE_HEADINGS):   # a provenance claim -> police THIS line
            out.append(_fix_line(line))
            police_bullets = True                         # and any cue bullets immediately below
            continue
        if police_bullets and is_bullet and not _EXPLAIN_LABEL_RE.match(line):
            out.append(_fix_line(line))
            continue
        police_bullets = False                            # any non-bullet line closes the run
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# The bounded loop
# --------------------------------------------------------------------------- #

def run_agent(query: str, max_tool_calls: int = MAX_TOOL_CALLS, source: str = "",
              prior_scratch: list | None = None) -> dict:
    """Run one turn. Returns {answer, trajectory, asked, drafts, scratch}.

    Multi-turn (task #3): when the agent asks a clarifying question, the app stores the
    returned `scratch` and passes it back as `prior_scratch` with the learner's reply, so the
    next turn CONTINUES from the same context (it remembers what it ingested / extracted)
    instead of starting over. There is NO cap on the number of reply rounds; only each turn's
    tool calls are bounded by max_tool_calls.

    `drafts` collects any enrich() output so the app can route it to HITL review.
    `source` (optional): a local .srt/video/audio path the learner attached.
    """
    if not config.has_ai_key():
        return {"answer": "No AI key configured (.env). The agent loop needs one; "
                          "you can still use the intent buttons for deterministic steps.",
                "trajectory": [], "asked": False, "drafts": [],
                "scratch": list(prior_scratch or [])}

    system = _system_prompt()
    cont_rem, cont_source = [], ""      # askfix REBASE: deterministic "explain the rest" (see below)
    cont_stage = []                    # askfix S19 (A1): deterministic "save the rest to queue"
    cont_done = False                  # askfix S19 (A2): continuation asked but nothing left
    if prior_scratch:
        scratch = list(prior_scratch) + [f"USER REPLY: {query}"]

        def _remaining_from_scratch():
            rem_line = next((ln for ln in reversed(prior_scratch)
                             if isinstance(ln, str)
                             and ln.startswith("REMAINING_UNEXPLAINED:")), "")
            terms = [t.strip() for t in rem_line.split(":", 1)[1].split(",")] if rem_line else []
            return rem_line, [t for t in terms if t][:8]

        def _source_from_scratch():
            src_line = next((ln for ln in prior_scratch
                             if isinstance(ln, str)
                             and ln.startswith("ATTACHED SOURCE FILE:")), "")
            return src_line.split(":", 1)[1].strip().splitlines()[0].strip() if src_line else ""

        # askfix S19 (A1): a SAVE intent ("lưu/save/stage", or "ghi/thêm ... vào queue") must be
        # detected BEFORE the continuation regex — the old order let "ghi các từ còn lại vào queue"
        # match "còn lại" and get force-EXPLAINED instead of STAGED. When it's a save request we
        # stage the remaining terms deterministically (below); we never route it into explain.
        _stage_intent = re.search(
            r"\b(lưu|save|stage)\b|\b(ghi|thêm|add|put)\b.*\b(queue|hàng chờ|review|danh sách|list)\b",
            query, re.IGNORECASE)
        # S19 OPEN-1: a FIND/LOOK-UP request ("tìm câu cho các cụm sau: …", "find the line for …")
        # must NOT be swallowed by the continuation regex just because it contains a NOUN like
        # "các cụm" / "the rest". Detect a search verb + a script/line noun and skip continuation
        # so the query routes to the LLM (or a future find-line button) instead of "all caught up".
        _find_intent = re.search(
            r"\b(tìm|tra|find|search|locate|look up)\b.*"
            r"\b(script|transcript|câu|sentence|line|dòng|thoại|phụ đề|subtitle)\b",
            query, re.IGNORECASE)
        if _stage_intent:
            # askfix S19 (#1 count fix): a BULK save is routed to Python so the exact set is staged
            # (the LLM path guessed a subset — "saved 14" of 28 — and mis-counted). "all/them/tất
            # cả" -> the FULL discovered set (ALL_FOUND); "còn lại/the rest" -> REMAINING. A specific
            # NAMED save ("lưu fire away và blend in") has neither cue -> left to the agent (the
            # learner just tells the AI which few to keep). NOT capped at 8 (that cap is for explain
            # batches); a save stages every requested term in one dedup'd stage_for_review call.
            def _full_marker(prefix):
                ln = next((l for l in reversed(prior_scratch)
                           if isinstance(l, str) and l.startswith(prefix)), "")
                return [t.strip() for t in ln.split(":", 1)[1].split(",") if t.strip()] if ln else []
            _all_found = _full_marker("ALL_FOUND:")
            _rem_full = _full_marker("REMAINING_UNEXPLAINED:")
            if re.search(r"còn lại|the rest|\brest\b|remaining|chưa (?:giải|lưu)", query, re.IGNORECASE):
                cont_stage = _rem_full[:40]
            elif re.search(r"\b(all|everything|them|those|tất cả|toàn bộ|hết|mọi|chúng)\b",
                           query, re.IGNORECASE):
                cont_stage = (_all_found or _rem_full)[:40]
            else:
                cont_stage = []            # specific/named save -> let the agent stage those terms
            cont_source = _source_from_scratch()
        # A continuation request ("explain the rest / giải thích thêm / tiếp...") is handled by
        # PYTHON DIRECTLY (Flash ignores text steers, and re-explained/ copy-pasted the old batch).
        # Here we only COMPUTE the remaining terms + the source; the pipeline runs before the loop.
        elif (not _find_intent) and re.search(
                       r"explain the rest|the rest|rest of|còn lại|giải thích thêm|giải thích tiếp"
                       r"|thêm các|tiếp theo|next batch|next ones|continue",
                       query, re.IGNORECASE):
            rem_line, cont_rem = _remaining_from_scratch()
            # askfix S19 (A2): marker PRESENT but empty = everything already explained. Do NOT
            # re-read a stale batch (the old bug: a never-reset REMAINING line was re-explained
            # forever). Signal a deterministic "all done" reply instead of re-explaining.
            cont_done = bool(rem_line) and not cont_rem
            cont_source = _source_from_scratch()
    else:
        scratch = [f"USER QUERY: {query}"]
        if source:
            scratch.append(
                f"ATTACHED SOURCE FILE: {source}\n"
                "To answer about this media, call ingest_transcript with EXACTLY this path, "
                "then call extract_vocab with its `transcript` argument set to the `srt_path` "
                "string ingest_transcript returned (NOT the transcript text — that is truncated "
                "in observations) to find its vocabulary, then wordnet_lookup / recall the words "
                "you will discuss, and explain LAST.")
    trajectory, drafts = [], []
    ingested_source = source          # S18 #6: source whose transcript is cached this turn
    # askfix REBASE: ONE explain per turn ends the turn (S16 frame); the deterministic
    # continuation above may also produce one. Kept as a list so _finish (join/renumber/
    # dedup) has a single shape for both paths.
    explain_outputs = []
    extracted_cands = []               # S18 HEART §1a: extract_vocab candidates (term -> real cue)
    recalled_known = False             # S18 (owner B): a recall hit fired this turn
    # S18 askfix (fix 2): accumulate REAL source lines seen this turn so the fabrication guard
    # works on recall-only follow-up turns too (not just when a transcript is ingested).
    turn_cues = {}                     # normalized term -> its real source line (for replacement)
    turn_real_lines = set()            # every real occurrence/candidate sentence seen this turn
    turn_sources = set()               # film basenames whose cached transcript we can check
    if source:
        turn_sources.add(source)
    # askfix D3: a follow-up turn usually has no `source` arg — recover the film from the
    # scratch's ATTACHED SOURCE FILE line so transcript-backed checks (recall-miss NOTE,
    # quote guard, explain materials) still know which cached transcript to consult.
    if prior_scratch and not ingested_source:
        _src = next((ln for ln in prior_scratch
                     if isinstance(ln, str) and ln.startswith("ATTACHED SOURCE FILE:")), "")
        _src = _src.split(":", 1)[1].strip().splitlines()[0].strip() if _src else ""
        if _src:
            ingested_source = _src
            turn_sources.add(_src)
    # askfix (owner #1): deterministic end-of-turn summary for a discovery turn — Python owns it
    # (0 extra AI calls), so it is always accurate about counts and never fabricates.
    explained_terms = []               # terms actually run through `explain` this turn
    known_terms = {}                   # lower term -> where it's already saved ("graph"/"queue")
    related_links = {}                 # S19: new term (lower) -> {existing graph terms it links to}

    def _harvest_recall(res, qterm=None):
        """Pull (term, sentence, source) from a recall result (single or batch shape) into the
        turn's real-line ground truth — so a fabricated 'From your graph/video' quote about an
        already-learned word is caught even when nothing was ingested this turn. `qterm` (batch
        key) lets us note associative CROSS-graph links: a discovered word that is a target inside
        an existing node's edges (synonym/is_a/…) -> surfaced in the summary's 🔗 block."""
        if not isinstance(res, dict):
            return
        batch = res.get("batch")
        if isinstance(batch, dict):
            for t, r in batch.items():
                _harvest_recall(r, qterm=t)
            return
        # S19: associative link — this queried word relates to EXISTING graph nodes (as_related =
        # keys of nodes whose edges point at it). Skip when the word is itself an exact node
        # (already covered by the "Already LEARNED" bucket) to avoid double-reporting.
        if qterm:
            ql = str(qterm).strip().lower()
            for rk in (res.get("as_related") or []):
                # S19 BUG-2: keep the FULL node key (lemma#sense) so the summary can look up the
                # actual edge type + node meaning — matching by lemma alone hid sense-mismatches.
                rkey = (str(rk) if isinstance(rk, str)
                        else str(rk.get("key", "")) if isinstance(rk, dict) else "")
                rt = rkey.split("#")[0].strip().lower()
                if rt and rt != ql and ql not in known_terms:
                    related_links.setdefault(ql, set()).add(rkey)
        if res.get("found") and isinstance(res.get("as_main_node"), dict):
            kt = str(res["as_main_node"].get("term", "") or "").strip().lower()
            if kt:
                known_terms.setdefault(kt, "your collection")   # committed graph node
        for e in (res.get("in_review_queue") or []):            # staged, awaiting approval
            k = str(e.get("key", "")).split("#")[0].strip().lower() if isinstance(e, dict) else ""
            if k:
                known_terms.setdefault(k, "review queue")
        main = res.get("as_main_node")
        if isinstance(main, dict):
            term = str(main.get("term", "") or "")
            for occ in (main.get("occurrences") or []):
                if not isinstance(occ, dict):
                    continue
                sent = str(occ.get("sentence", "") or "").strip()
                if sent:
                    turn_real_lines.add(sent)
                    if term:
                        turn_cues.setdefault(term.strip().lower(), sent)
                if occ.get("source"):
                    turn_sources.add(str(occ["source"]))
        for h in (res.get("in_sentences") or []):
            if isinstance(h, dict) and str(h.get("sentence", "") or "").strip():
                turn_real_lines.add(str(h["sentence"]).strip())
                if h.get("source"):
                    turn_sources.add(str(h["source"]))

    def _finish(answer: str, asked: bool):
        """Terminal return: if explain produced text this turn, the learner-facing answer is
        the FULL concatenation of every batch (verbatim). A trailing ask/final is appended so a
        follow-up question still shows, but the explanations are never dropped.

        S18 HEART §1e: the model's `final` often RE-PRINTS the batches it just explained (Q2
        dumped the whole answer twice; Q1 recapped all 15 terms at the tail). Drop that trailing
        recap — if most of its substantive sentences already appear in the joined explanations —
        so each phrase shows exactly ONCE. A genuinely new follow-up (low overlap) is kept."""
        if explain_outputs:
            joined = "\n\n---\n\n".join(explain_outputs)
            joined = _format_explain_fields(joined)   # askfix S19 (#2): un-glue field labels
            if len(explain_outputs) > 1:       # owner #5: continuous 1..N across chained batches
                joined = _renumber_terms(joined)
            answer = f"{joined}\n\n{extra}" if (extra := _dedup_final(joined, answer)) else joined
            # askfix (owner #1): Python OWNS the end-of-turn summary (0 AI, always accurate).
            if asked is False and "### 📋" not in answer and "Review it:" not in answer:
                summary = _summary_block(extracted_cands, explained_terms, known_terms, related_links)
                if summary:
                    answer += summary
            # askfix (owner V2.2): persist the un-explained remainder INTO the scratch so the
            # NEXT turn ("explain the rest") can be steered deterministically — live test showed
            # the model otherwise re-explains the same batch or just asks back.
            _exp = {_norm_loc(t) for t in explained_terms}
            _rem = [c.get("term") for c in extracted_cands
                    if c.get("term") and _norm_loc(c["term"]) not in _exp
                    and str(c["term"]).lower() not in known_terms]
            if not _rem:
                # continuation turn (no fresh extract): carry the PREVIOUS remainder forward,
                # minus what this turn just explained, so a 3rd "explain more" keeps advancing.
                prev = next((ln for ln in reversed(scratch)
                             if isinstance(ln, str) and ln.startswith("REMAINING_UNEXPLAINED:")), "")
                _rem = [t.strip() for t in prev.split(":", 1)[1].split(",")
                        if t.strip() and _norm_loc(t) not in _exp] if prev else []
            # askfix S19 (A2): ALWAYS refresh the marker — write it EMPTY when nothing remains so
            # a stale REMAINING line from an earlier turn can never be re-read and re-explained
            # (the old code only wrote it when non-empty, leaving the last batch to loop forever).
            scratch.append("REMAINING_UNEXPLAINED: " + ", ".join(dict.fromkeys(_rem)))
            # askfix S19 (A3): everything found was explained this turn -> offer the next step
            # (save to queue) so the answer never dead-ends. Discovery-with-remainder is already
            # handled by _summary_block; only add this when there is genuinely nothing left.
            if not _rem and asked is False and "👉 Next" not in answer:
                answer += _next_step_hint("explained_all")
        # askfix S19 (#1): persist the FULL discovered set (once, on the discovery turn) so a
        # "save all" — from the button or a typed request — stages every found phrase
        # deterministically instead of an LLM-guessed subset. Additive scratch line; it carries
        # forward untouched on later turns (scratch always includes prior_scratch).
        if extracted_cands and not any(
                isinstance(s, str) and s.startswith("ALL_FOUND:") for s in scratch):
            _allf = list(dict.fromkeys(c.get("term") for c in extracted_cands if c.get("term")))
            if _allf:
                scratch.append("ALL_FOUND: " + ", ".join(_allf))
        return {"answer": answer, "trajectory": trajectory,
                "asked": asked, "drafts": drafts, "scratch": scratch}

    # askfix S19 (A2): a continuation request when NOTHING remains -> deterministic "all done"
    # reply (0 AI, no re-explain). This is the other half of the anti-loop fix: even if a stale
    # marker somehow survived, an empty remainder can never be re-explained.
    if cont_done and not cont_rem:
        return {"answer": "You're all caught up — every phrase I found has been explained."
                          + _next_step_hint("explained_all"),
                "trajectory": [], "asked": False, "drafts": [], "scratch": scratch}

    # askfix S19 (A1) — DETERMINISTIC "save to queue": the learner asked to SAVE (not explain) the
    # remaining phrases. Python stages them directly via stage_for_review, grounding each term to
    # its verbatim cue (recall + _materials_for, the same cues explain uses) so staged rows carry
    # real source lines (+ timestamps when an srt is cached). A "save"-worded request can never be
    # hijacked into re-explaining. REMAINING is left intact so the learner can still "explain the
    # rest" afterward. Falls through to the normal loop if nothing grounds / anything goes wrong.
    if cont_stage:
        if cont_source:
            ingested_source = cont_source
            turn_sources.add(cont_source)
        batch = ", ".join(cont_stage)
        try:
            rc = call_tool("recall", {"lemma": batch})
            trajectory.append({"tool": "recall", "args": {"lemma": batch},
                               "thought": "save-to-queue: recall remaining terms for grounding"})
            if isinstance(rc, dict):
                _harvest_recall(rc)
            cues = dict(turn_cues)
            mats = _materials_for(batch, ingested_source, extracted_cands) if ingested_source else None
            for h in ((mats or {}).get("hits") or []):
                if h.get("term") and h.get("line"):
                    cues[str(h["term"]).strip().lower()] = h["line"]
            sentences = {t: cues[t.strip().lower()] for t in cont_stage
                         if cues.get(t.strip().lower())}
            res = call_tool("stage_for_review",
                            {"terms": cont_stage, "sentences": sentences,
                             "source": ingested_source or ""})
            trajectory.append({"tool": "stage_for_review", "args": {"terms": cont_stage},
                               "thought": "save-to-queue: stage remaining terms"})
            if isinstance(res, dict):
                newly = res.get("newly_staged", []) or []
                already = res.get("already_present", []) or []
                ung = res.get("ungrounded", []) or []
                parts = []
                if newly:
                    parts.append(f"Saved **{len(newly)}** phrase(s) to your review queue: "
                                 f"{', '.join(map(str, newly))}.")
                if already:
                    parts.append(f"{len(already)} were already in the queue, left unchanged: "
                                 f"{', '.join(map(str, already))}.")
                for u in ung:
                    if isinstance(u, dict) and u.get("term"):
                        # S19: surface the flag REASON here too — it carries the suggested real
                        # transcript line (_transcript_hint), so the learner sees the fix inline
                        # instead of only in the Review tab (the LLM stage path already does this).
                        _r = str(u.get("reason", "")).strip()
                        parts.append(f"⚠ \"{u['term']}\" was staged but flagged ungrounded"
                                     + (f" — {_r}" if _r else "")
                                     + " (fix its sentence in the Review tab before it can commit).")
                if not parts:
                    parts.append("Nothing new to save — those phrases are already in your queue.")
                # S19 OPEN-2: one block per Saved/⚠ line (was " ".join -> a single unreadable wall,
                # esp. with several ungrounded ⚠). Learner-facing only; the model OBSERVATION below
                # keeps its own wording. _next_step_hint already starts on its own line.
                ans = "\n\n".join(parts) + _next_step_hint("staged")
                scratch.append(f"ACTION: stage_for_review({json.dumps(cont_stage, ensure_ascii=False)})")
                scratch.append("OBSERVATION: remaining phrases staged to the review queue.")
                return {"answer": ans, "trajectory": trajectory, "asked": False,
                        "drafts": drafts, "scratch": scratch}
        except Exception:
            pass  # fall through to the normal loop if anything goes wrong

    # askfix REBASE — DETERMINISTIC "explain the rest": run recall -> wordnet_lookup -> explain on
    # the remaining terms DIRECTLY (no LLM decision), then finish. This is the heart guarantee:
    # a continuation can never call the wrong tool or copy the old batch, because Python drives it.
    if cont_rem:
        batch = ", ".join(cont_rem)
        if cont_source:
            ingested_source = cont_source
            turn_sources.add(cont_source)
        try:
            rc = call_tool("recall", {"lemma": batch})
            trajectory.append({"tool": "recall", "args": {"lemma": batch},
                               "thought": "continuation: recall remaining terms"})
            if isinstance(rc, dict):
                _harvest_recall(rc)
            try:
                call_tool("wordnet_lookup", {"term": batch})
                trajectory.append({"tool": "wordnet_lookup", "args": {"term": batch},
                                   "thought": "continuation: wordnet for remaining terms"})
            except Exception:
                pass
            ctx = rc if isinstance(rc, dict) else {}
            mats = _materials_for(batch, ingested_source, extracted_cands) if ingested_source else None
            if mats:
                ctx = dict(ctx) if isinstance(ctx, dict) else {}
                ctx["materials"] = mats
            res = call_tool("explain", {"query": batch, "context": ctx})
            trajectory.append({"tool": "explain", "args": {"query": batch},
                               "thought": "continuation: explain remaining terms"})
            # 429-safe: explain degrades to an error STRING (never raises) after its own
            # backoff retries are spent — _explain_ok rejects it, so we fall through to the
            # normal loop WITHOUT marking the terms explained (REMAINING_UNEXPLAINED intact;
            # the learner can simply say "explain the rest" again once quota recovers).
            if _explain_ok(res):
                cues = dict(turn_cues)
                for h in ((mats or {}).get("hits") or []):
                    if h.get("term") and h.get("line"):
                        cues[str(h["term"]).strip().lower()] = h["line"]
                res = _verify_video_quotes(res, turn_sources, cues, turn_real_lines)
                explain_outputs.append(res)
                explained_terms.extend(cont_rem)
                scratch.append(f"ACTION: explain({json.dumps(batch, ensure_ascii=False)})")
                scratch.append("OBSERVATION: remaining phrases explained (shown to the learner).")
                return _finish("", asked=False)
        except Exception:
            pass  # fall through to the normal loop if anything goes wrong

    for step in range(max_tool_calls + 1):
        hint = ("\n\nDecide the next single step as STRICT JSON." if step < max_tool_calls
                else "\n\nYou have reached the tool-call cap. Respond with a final JSON answer now."
                     " If you could not explain every requested phrase within the budget, say so"
                     " explicitly and list which terms still need explaining.")
        prompt = "\n".join(scratch) + hint
        try:
            raw = call_ai(prompt, system)
            decision = _parse_decision(raw)
        except Exception as e:
            return {"answer": f"(agent error: {e})", "trajectory": trajectory,
                    "asked": False, "drafts": drafts, "scratch": scratch}

        if "ask_user" in decision:
            scratch.append(f"AGENT ASKED: {decision['ask_user']}")
            return _finish(decision["ask_user"], asked=True)
        if "final" in decision:
            return _finish(_unwrap_final(decision["final"]), asked=False)

        action = decision.get("action") or {}
        tool, targs = action.get("tool"), action.get("args") or {}
        if step >= max_tool_calls:
            # Tool budget exhausted — the model still tried an action; do NOT execute it
            # (otherwise cap=8 quietly allows a 9th call). Deterministic skip, loop ends.
            scratch.append("OBSERVATION: tool budget exhausted — answer now from what you have.")
            continue
        if tool not in TOOLS:
            scratch.append(f"OBSERVATION: unknown tool {tool!r}. Choose from: {', '.join(TOOLS)}")
            continue
        # S18 #6: remember the transcript source the agent ingested, and inject the "in your
        # materials" grounding into explain's context so its answer separates FILM vocab from
        # extra knowledge. Deterministic; only augments (never overwrites the model's context).
        if tool == "ingest_transcript" and isinstance(targs.get("source"), str):
            ingested_source = targs["source"]
            turn_sources.add(targs["source"])
        mats = None
        if tool == "explain" and ingested_source:
            # HEART §1a: ground on the terms ACTUALLY being explained — reusing the real cues
            # extract_vocab discovered this turn (a phrase not in `query` still gets its line).
            # FEATURE-3 (lazy): only when the learner asked for the in-scene meaning do we pull
            # the ±1 neighbour context + tell explain to interpret the word AS USED in the scene.
            want_scene = bool(_IN_SCENE_RE.search(query or ""))
            mats = _materials_for(str(targs.get("query", "")), ingested_source, extracted_cands,
                                  with_scene=want_scene)
            if mats:
                ctx = targs.get("context")
                if not isinstance(ctx, dict):
                    ctx = {} if ctx is None else {"_prev": ctx}
                ctx.setdefault("materials", mats)
                if want_scene:
                    ctx["explain_in_scene"] = True
                targs["context"] = ctx
        try:
            result = call_tool(tool, targs)
            if tool == "enrich" and isinstance(result, list):
                drafts.extend(result)
            if tool == "recall" and isinstance(result, dict):
                if result.get("found"):
                    recalled_known = True      # S18 (owner B): drives the review tip in _finish
                # S19-S4: pass the queried lemma so a SINGLE-word recall (not a batch) still
                # harvests cross-graph links (related_links). A batch-shaped result recurses with
                # its own per-term keys inside _harvest_recall, so this qterm is used only for the
                # single-term shape — no effect on the batch/discovery path.
                _harvest_recall(result, qterm=targs.get("lemma"))  # fix 2: real graph lines guard
            # HEART §1a: keep each discovered candidate's VERBATIM source cue so a later explain
            # can quote the real line for a phrase the miner found (not just query-listed terms).
            if tool == "extract_vocab" and isinstance(result, list):
                for c in result:
                    if isinstance(c, dict) and str(c.get("term", "")).strip() \
                            and str(c.get("sentence", "")).strip():
                        extracted_cands.append({"term": c["term"], "surface": c.get("surface", ""),
                                                "sentence": c["sentence"]})
                        turn_real_lines.add(str(c["sentence"]).strip())
                        turn_cues.setdefault(str(c["term"]).strip().lower(), str(c["sentence"]).strip())
                        if c.get("surface"):
                            turn_cues.setdefault(str(c["surface"]).strip().lower(), str(c["sentence"]).strip())
            # HEART §1b (fix 2): Python is the last word on every quote attributed to a real source
            # (From this video / From your graph) — using this turn's real lines (ingest OR recall)
            # as ground truth. Replace any fabricated line with the real cue / "Not found…" (0 AI).
            if tool == "explain" and isinstance(result, str) and result.strip():
                cues = dict(turn_cues)
                for h in ((mats or {}).get("hits") or []):
                    if h.get("term") and h.get("line"):
                        cues[str(h["term"]).strip().lower()] = h["line"]
                result = _verify_video_quotes(result, turn_sources, cues, turn_real_lines)
        except Exception as e:           # tool error -> tell the model, keep going (no crash)
            result = {"error": str(e)}
        trajectory.append({"tool": tool, "args": targs,
                           "thought": decision.get("thought", "")})   # show WHY (demo clarity)
        # askfix REBASE (return to S16 behavior): explain ENDS the turn immediately. Its text IS
        # the learner-facing answer, so returning now (a) drops the extra decision call that let
        # the model chain/repeat, and (b) keeps the full explanation OUT of scratch — so a later
        # "explain the rest" turn has nothing to copy-paste and must call explain on fresh terms.
        # _finish still appends the deterministic found/known/remaining summary (+ REMAINING marker).
        if tool == "explain" and _explain_ok(result):
            explain_outputs.append(result)
            for t in str(targs.get("query", "")).split(","):
                if t.strip():
                    explained_terms.append(t.strip())
            scratch.append(f"THOUGHT: {decision.get('thought','')}")
            scratch.append(f"ACTION: explain({json.dumps(targs.get('query', ''), ensure_ascii=False)})")
            scratch.append("OBSERVATION: explanation delivered to the learner (shown in full to them; "
                           "not repeated here).")
            return _finish("", asked=False)
        if tool == "stage_for_review" and isinstance(result, dict):
            # Report NEW vs ALREADY-PRESENT separately. stage_for_review dedups by key and KEEPS
            # an existing row's status, so re-staging an already-reviewed word adds NO new
            # "to review" row. The agent must NOT tell the learner "saved N" when most were
            # duplicates already in the queue — otherwise they look for them under "to review",
            # don't find them (they're already approved), and think the save failed.
            newly = result.get("newly_staged", [])
            already = result.get("already_present", [])
            updated = result.get("updated", [])            # S18 #1/#4: re-staged w/ a fixed sentence
            parts = []
            if newly:
                parts.append(f"Added {len(newly)} NEW word(s) to the review queue (blank status, "
                             f"awaiting review): {', '.join(map(str, newly))}.")
            if updated:
                parts.append(f"UPDATED the source sentence of {len(updated)} word(s) already in the "
                             f"queue (they were flagged ungrounded; the corrected line is now in the "
                             f"review table and the flag is cleared): {', '.join(map(str, updated))}.")
            if already:
                parts.append(f"{len(already)} word(s) were ALREADY in the queue and left UNCHANGED "
                             f"(NOT re-added, sentence NOT modified): {', '.join(map(str, already))}. "
                             f"Do NOT claim you updated these — if the learner wants their sentence "
                             f"changed, either give the exact transcript line so it re-grounds, or "
                             f"tell them to edit the 'sentence' cell for that row in the review table.")
            # S17 (was S16 hard-drop): an ungrounded term IS staged but FLAGGED — the learner
            # decides in review. Tell the model so it reports the flag honestly and offers to
            # re-check the transcript for the real line.
            ungrounded = result.get("ungrounded", []) or []
            for u in ungrounded:
                if isinstance(u, dict) and u.get("term"):
                    parts.append(f"STAGED but FLAGGED ungrounded: {u['term']} — "
                                 f"{u.get('reason', 'no source sentence')} (it appears in the "
                                 "review table flagged '⚠ ungrounded'; the learner must supply "
                                 "the real source line before it can commit)")
            if not parts:
                parts.append("No words were staged.")
            obs = (" ".join(parts) + " Report this to the learner EXACTLY — do NOT claim a word "
                   "was newly saved if it was already present.")
        else:
            obs = _observe(result)
        # askfix D3 (FIX B): a recall MISS must not become "it's not in the script" — Python
        # checks the cached transcript itself and appends the verbatim cue when the term IS there.
        if tool == "recall" and isinstance(result, dict):
            obs += _recall_miss_note(result, targs, turn_sources)
        scratch.append(f"THOUGHT: {decision.get('thought','')}")
        scratch.append(f"ACTION: {tool}({json.dumps(targs, ensure_ascii=False)})")
        scratch.append(f"OBSERVATION: {obs}")

    return _finish("(reached step cap without a final answer)", asked=False)


# --------------------------------------------------------------------------- #
# Final artefacts — generated ONLY after HITL approval (GATE-EXPORT, task #1)
# --------------------------------------------------------------------------- #

def build_final_exports(drafts: list, run_id: str = "committed", srt_path: str = "",
                        gen_dictation: bool = True) -> dict:
    """Generate the learner-facing deliverables from APPROVED drafts (Anki deck, Obsidian
    vault, provenance infolog, highlighted .ass). Called by app.commit_approved AFTER the
    human approves — never during Mine (HITL: nothing final is produced from unapproved
    drafts). Each export is independently no-crash; a failing one yields "".

    `drafts` = list of {"node": <Node dict>, "surface": str, "clip": {...}?} for approved items.
    Returns {"deck", "obsidian_vault", "infolog", "highlighted_ass"}.
    """
    nodes = [d.get("node", {}) for d in (drafts or []) if isinstance(d, dict) and d.get("node")]
    out = {"deck": {}, "obsidian_vault": "", "infolog": "", "highlighted_ass": ""}
    if not nodes:
        return out

    try:
        out["deck"] = call_tool("make_anki", {"units": drafts, "run_id": run_id,
                                              "gen_dictation": gen_dictation}) or {}
    except Exception:
        out["deck"] = {}
    try:
        from obsidian_export import export_obsidian
        out["obsidian_vault"] = export_obsidian(nodes, run_id=run_id)
    except Exception:
        out["obsidian_vault"] = ""
    try:
        from infolog_export import export_infolog
        from _common import run_dir, load_graph, GRAPH_PATH
        # Infolog = the CUMULATIVE ledger of the whole committed graph (S14 T10) — the caller
        # (app.commit_approved) invokes this AFTER save_graph, so the graph already includes
        # this batch. Deck/Obsidian/.ass stay batch-scoped above.
        graph = load_graph(GRAPH_PATH)
        all_nodes = [n.model_dump() for n in graph.nodes.values()]
        graph_keys = {n.get("key") for n in all_nodes}
        # batch nodes not (yet) in the graph are appended so nothing approved is ever missing
        all_nodes += [n for n in nodes if n.get("key") not in graph_keys]
        out["infolog"] = export_infolog(
            all_nodes, os.path.join(run_dir(run_id), "infolog.txt"))["out"]
    except Exception:
        out["infolog"] = ""
    if srt_path and os.path.exists(srt_path):
        try:
            from subtitle_highlight import export_highlighted_ass
            from _common import run_dir
            terms = [d.get("surface") or d.get("node", {}).get("term", "") for d in drafts]
            out["highlighted_ass"] = export_highlighted_ass(
                srt_path, terms, os.path.join(run_dir(run_id), "highlighted.ass"))["out"]
        except Exception:
            out["highlighted_ass"] = ""
    return out


# --------------------------------------------------------------------------- #
# Fallback: deterministic intent runners (button-driven — HANDOVER §6)
# --------------------------------------------------------------------------- #

def run_intent(intent: str, **kw) -> dict:
    """Fixed tool sequences per intent (LLM still does enrich/explain inside).

    intents: "mine" (source=, focus=), "explain" (query=), "expand" (term=).
    Returns a dict with the artifacts produced (drafts/graph/apkg/answer).
    """
    if intent == "mine":
        source, focus = kw.get("source", ""), kw.get("focus", "")
        media = kw.get("media", "")          # optional companion video -> Anki audio+screenshot
        ing = call_tool("ingest_transcript", {"source": source})
        cands = call_tool("extract_vocab", {"transcript": ing, "focus": focus})
        units = []
        for c in cands:
            recall_hit = call_tool("recall", {"lemma": c["term"]})
            senses = call_tool("wordnet_lookup", {"term": c["term"]})["senses"]
            # B1: query ConceptNet for the life-context layer (used_for/has_context) +
            # OOV part_of. Per-term + sequential; any failure is non-fatal (no-crash).
            cn_edges = []
            if config.CONCEPTNET_PER_TERM:
                try:
                    cn_edges = call_tool("conceptnet_lookup", {"term": c["term"]}).get("edges", [])
                except Exception:
                    cn_edges = []
            st, en = _locate_timestamp(ing.get("segments", []),
                                       c.get("surface") or c["term"], c["sentence"])
            # S18 P0-1: `st`/`en` are now FLOAT seconds for timed sources. Keep the
            # occurrence provenance as "HH:MM:SS" strings (schema.Occurrence.start is str),
            # but stash the raw float in start_sec/end_sec so the clip keeps millisecond
            # precision (the occurrence's rounded HH:MM:SS is display-only).
            units.append({"term": c["term"], "sentence": c["sentence"],
                          "surface": c.get("surface", ""), "tag": c.get("tag", ""),
                          "senses": senses, "cn_edges": cn_edges,
                          "start": _hms(st), "end": _hms(en),   # provenance: HH:MM:SS
                          "start_sec": st if isinstance(st, (int, float)) and not isinstance(st, bool) else "",
                          "end_sec": en if isinstance(en, (int, float)) and not isinstance(en, bool) else "",
                          "source": os.path.basename(source), "recalled": recall_hit["found"]})
        drafts = call_tool("enrich", {"units": units, "source": os.path.basename(source), "focus": focus})
        nodes = [d["node"] for d in drafts]
        # Note EVERY occurrence: extract_vocab dedups a term to ONE candidate, but a word
        # recurs (often in different inflections). _collect_repeats scans the segments and
        # appends an Occurrence for each OTHER line, recording that line's ORIGINAL surface
        # form. The enrich occurrence (which carries lemmas) stays primary; cap per node so a
        # very common word doesn't bloat the graph. -> infolog/graph note all sightings.
        segs = ing.get("segments", [])
        for d in drafts:
            _collect_repeats(d["node"], segs)
        # Optional companion video: attach a {video,start,end} clip per draft so make_anki
        # can cut SHORT audio + a screenshot for each card (reuses make_anki's existing clip
        # support — no new tool). Each draft's primary occurrence already carries the source
        # start/end (provenance). Additive: with no media (the default) cards stay text-only.
        if media and os.path.exists(media):
            # S18 P0-1: carry the FLOAT segment seconds (unit.start_sec/end_sec) into the
            # clip so make_anki cuts with millisecond precision + padding (not the rounded
            # HH:MM:SS provenance). enrich preserves unit order, so units[i] <-> drafts[i].
            for u, d in zip(units, drafts):
                cs, ce = u.get("start_sec"), u.get("end_sec")
                if isinstance(cs, (int, float)) and not isinstance(cs, bool) \
                        and isinstance(ce, (int, float)) and not isinstance(ce, bool) and ce > cs:
                    cs, ce = _expand_to_neighbor(segs, cs, ce)   # ±1 subtitle context
                    d["clip"] = {"video": media, "start": cs, "end": ce}
        # HITL GATE-EXPORT (task #1): Mine renders only a PREVIEW graph of the DRAFT nodes.
        # The final deliverables (Anki deck / Obsidian vault / infolog / highlighted .ass) are
        # NOT produced here — they are generated by build_final_exports() at Commit, from the
        # subset the human approved. Nothing final ever comes from unapproved drafts.
        graph = call_tool("build_render_graph", {"units": nodes, "run_id": ing["run_id"]}) if nodes else ""
        # Words the graph ALREADY knows (recall hit before enrich) — surfaced so the app can
        # tell the learner these are re-encounters, not new words (additive key).
        recalled_terms = [u["term"] for u in units if u.get("recalled")]
        return {"drafts": drafts, "graph": graph, "recalled_terms": recalled_terms,
                "run_id": ing["run_id"], "srt_path": ing.get("srt_path", "")}

    if intent == "explain":
        q = kw.get("query", "")
        hit = call_tool("recall", {"lemma": q})
        ans = call_tool("explain", {"query": q, "context": hit if hit["found"] else None})
        return {"answer": ans, "recall": hit}

    if intent == "expand":
        term = kw.get("term", "")
        hit = call_tool("recall", {"lemma": term})
        senses = call_tool("wordnet_lookup", {"term": term})["senses"]
        return {"recall": hit, "senses": senses}

    return {"error": f"unknown intent {intent!r}"}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "explain 'fed up'"
    print(json.dumps(run_agent(q), ensure_ascii=False, indent=2))
