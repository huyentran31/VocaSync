"""
extract_vocab.py — Tool #3 (read, medium cost, AI).

Mine candidate learning items {term, sentence} from a transcript. The LLM only
SELECTS spans that already exist in the transcript — it does not define or relate
them (that is wordnet_lookup + enrich downstream). Keeping extraction narrow is the
deterministic-first discipline (Day-4): AI proposes candidates, WordNet grounds them.

GĐ4 quality upgrade (HANDOVER S3 §3.4 / §5.1) — bounded multi-call + self-correct:
  1) gather pass(es): call the AI, union UNIQUE terms (drop ones already found) until
     we reach EXTRACT_MIN_UNIQUE or run out of the gather budget. AI often returns
     only 3-4 items in one shot; a second pass that excludes the found terms pulls more.
  2) Python self-correct (NO AI): flag any candidate whose surface/term does NOT
     literally appear in the transcript — i.e. a hallucinated/ungrounded pick. Mirrors
     legacy/file_utils._norm_match_local (normalize to [a-z0-9]+, substring check).
  3) one fix call: ask the AI to replace the flagged terms with items that DO appear.
Bounded to EXTRACT_MAX_CALLS total AI calls (default 3) via env knobs.

Reuse: legacy/ai_client.call_ai for the HTTP call + 429 backoff/retry. Parsing and
validation are local because this tool's contract ({term, sentence}) differs from the
old 8-field clip schema.

Error model (error_handling.gherkin):
  • no API key / provider error      → SystemError_ (HALT)
  • round-1 JSON unparseable          → retry once, then SystemError_
  • later gather/fix call unparseable → skip that round (we already have round-1 items)
  • a single malformed candidate item → skip it (clip-error), keep the rest
  • a single ungrounded candidate     → flag + try to fix, else drop (clip-error)
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from pydantic import BaseModel, ValidationError

import config
from ai_client import call_ai
from _common import SystemError_, log_tool_call

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"description": "Full text, or the segments list from ingest_transcript."},
        "focus": {"type": "string", "description": "Optional topic to bias selection (e.g. 'emotions')."},
        "max_terms": {"type": "integer", "description": "Cap candidates returned (default 20)."},
    },
    "required": ["transcript"],
}

# --- bounded multi-call knobs (env-overridable, no legacy edit) --------------- #
# Keep extraction cheap and bounded (AGENTS.md: bounded; HANDOVER §3.4 "~3 call").
_MIN_UNIQUE = int(os.getenv("EXTRACT_MIN_UNIQUE", "8"))   # gather until we have this many
_MAX_CALLS = int(os.getenv("EXTRACT_MAX_CALLS", "3"))      # hard cap on AI calls per run

# --- long-script coverage knobs (HANDOVER S8 §3.1) --------------------------- #
# A long transcript in ONE prompt gets uneven coverage: the model fixates on the
# salient/early lines and under-samples the tail. Above _CHUNK_LINES total lines we
# split the transcript into windows and gather from EACH (one AI call per window,
# excluding terms already found) so every region is sampled. Gated so short demo
# clips (the committed Charlie episodes) keep their single-pass behaviour + quota.
_CHUNK_LINES = int(os.getenv("EXTRACT_CHUNK_LINES", "500"))      # only chunk above this many lines
_LINES_PER_CHUNK = int(os.getenv("EXTRACT_LINES_PER_CHUNK", "250"))  # window size
_MAX_CHUNKS = int(os.getenv("EXTRACT_MAX_CHUNKS", "6"))         # cap windows (bound the AI calls)

# --- long-script QUOTA knobs (S12 T1) ---------------------------------------- #
# The caller's max_terms (default 20) is the right cap for a short demo clip, but a long
# film split into N windows should be allowed MORE total items — capping a feature-length
# transcript at 20 short-changes it. In CHUNKED mode only, the effective cap becomes
# min(_HARD_MAX, _PER_CHUNK * N_windows) so quota scales with length (floored at _MIN_UNIQUE).
# Short single-chunk runs are untouched (still bounded by max_terms, i.e. 8-20).
_PER_CHUNK = int(os.getenv("EXTRACT_PER_CHUNK", "10"))   # ~items to aim for per window
_HARD_MAX = int(os.getenv("EXTRACT_HARD_MAX", "48"))     # absolute ceiling on a single Mine run


class Candidate(BaseModel):
    """Validated extraction unit. Extra hints are optional and kept if present."""
    term: str                       # base/lemma form of the target item
    sentence: str                   # the exact line it appeared in
    surface: str = ""               # as-conjugated form in the sentence (optional)
    tag: str = ""                   # Phrasal Verb | Idiom | Collocation | Slang | Word


_SYSTEM_PROMPT = (
    "You are a vocabulary extraction engine for an English-learning tool. "
    "If a USER REQUEST/FOCUS is given, FOLLOW it: it may name a topic AND/OR restrict "
    "which item TYPES to pick (e.g. 'only phrasal verbs and fixed expressions' -> return "
    "only phrasal verbs and idioms). If NO request is given, fall back to selecting the "
    "most useful learning items (phrasal verbs, idioms, collocations, useful single words). "
    "ONLY pick items that literally appear in the transcript; never invent words. "
    "Return STRICT JSON: an array of objects with keys "
    '{"term","sentence","surface","tag"}. '
    '"term" = dictionary/base form (lemma): de-conjugate the verb in verbs, phrasal '
    'verbs AND collocations ("figured out" -> "figure out", "made a decision" -> '
    '"make a decision") and singularize nouns ("emissions" -> "emission"); keep idioms '
    'in their conventional dictionary form. Put the ORIGINAL conjugated form in "surface". '
    '"sentence" = the exact transcript line; '
    '"surface" = the form as it appears (may equal term); '
    '"tag" = one of Phrasal Verb, Idiom, Collocation, Slang, Word. '
    "Output ONLY the JSON array — no markdown, no commentary."
)


# --- optional deterministic TYPE filter ------------------------------------- #
# The free-text focus may request specific item TYPES ("only phrasal verbs and fixed
# expressions"). The prompt already biases toward them; this is the hard guarantee — a
# candidate whose `tag` is not a requested type is dropped (no extra AI call; uses the
# tag the model already assigned). Conservative keywords so a TOPIC like "expressions of
# emotion" doesn't accidentally restrict types.
_TYPE_KEYWORDS = {
    "phrasal_verb": ("phrasal verb", "phrasal-verb", "phrasal"),
    "idiom": ("idiom", "fixed expression", "fixed-expression"),
    "collocation": ("collocation",),
    "slang": ("slang",),
    "word": ("single word", "single-word"),
}


def _requested_types(focus: str) -> set:
    """Normalized word_types the focus explicitly asks for; empty = no type restriction."""
    f = (focus or "").lower()
    return {wt for wt, kws in _TYPE_KEYWORDS.items() if any(k in f for k in kws)}


def _tag_norm(tag: str) -> str:
    return (tag or "").strip().lower().replace(" ", "_")


def _as_text(transcript) -> str:
    """Accept ingest_transcript's segment list, a raw string, OR a path to an .srt file.

    The agent path passes only the `srt_path` string between tools (it cannot thread the
    full transcript object through compacted observations), so a bare .srt path must be
    read and stripped to plain text here — otherwise the path itself becomes 'the
    transcript' and nothing is extracted.
    """
    if isinstance(transcript, str):
        if transcript.lower().endswith(".srt") and os.path.exists(transcript):
            try:
                with open(transcript, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                return "\n".join(s for ln in raw.splitlines()
                                 if (s := ln.strip()) and not s.isdigit() and "-->" not in s)
            except Exception:
                return transcript
        return transcript
    if isinstance(transcript, dict) and "segments" in transcript:
        transcript = transcript["segments"]
    if isinstance(transcript, list):
        return "\n".join(
            (s.get("text", "") if isinstance(s, dict) else str(s)) for s in transcript
        )
    return str(transcript)


def _norm_match(s: str) -> str:
    """Lowercase + keep only [a-z0-9] tokens, single-spaced.

    Mirrors legacy/file_utils._norm_match_local so "ran into" matches a transcript
    line "I ran into him." regardless of punctuation/case.
    """
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


@lru_cache(maxsize=20000)
def lemmatize_term(term: str) -> str:
    """Canonicalize a term to its base/lemma form — DETERMINISTIC (WordNet morphy), no AI.

    askfix (owner V2.1 speed): memoized — grounding lemmatizes the SAME transcript tokens
    hundreds of times per turn (once per candidate × every SRT line). Pure function of `term`,
    so caching is safe and cuts the deterministic grounding cost several-fold.

    This is the single point that makes the personal graph dedup correctly: the Node.key
    is `lemma#sense`, so "emissions" and "emission" (or "figured out" and "figure out")
    must collapse to the SAME term or they become duplicate nodes within one session.

      • single word : wn.morphy("emissions") -> "emission" (only changes KNOWN inflections;
                       unknown words like "Paris" return None -> unchanged, so it is safe).
      • multi-word  : lemmatize the FIRST token as a verb ("figured out" -> "figure out")
                       but ACCEPT only if the result is a real WordNet phrase. This gate
                       both confirms phrasal verbs (figure_out/run_into/give_up exist) AND
                       protects non-verb phrases ("United States" -> "unite states" has no
                       synset, so it is rejected and the original is kept).

    Division of labour: the AI (step 1, extract prompt) already base-forms verbs, phrasal
    verbs and collocations; THIS function (step 2) is the deterministic ENFORCER for what
    WordNet can verify — it overrides the AI for single words / WordNet phrases and otherwise
    leaves the AI's lemma untouched (collocations like "make a decision" have no WordNet
    synset, so the AI's base form stands — and HITL review still gates it).

    No-crash: if WordNet is unavailable, return the lowercased original unchanged.
    """
    raw = (term or "").strip()
    if not raw:
        return raw
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return raw.lower()
    words = raw.lower().split()
    try:
        if len(words) == 1:
            return wn.morphy(words[0]) or words[0]
        lemma0 = wn.morphy(words[0], wn.VERB)
        if lemma0 and lemma0 != words[0]:
            cand = " ".join([lemma0] + words[1:])
            if wn.synsets(cand.replace(" ", "_")):   # accept only a real WordNet phrase
                return cand
    except Exception:
        return raw.lower()
    return " ".join(words)


def _split_into_chunks(text: str) -> list[str]:
    """Split a transcript into line-windows for even coverage of a LONG script.

    Returns [text] unchanged when the transcript is at/under _CHUNK_LINES (the common
    demo case). Otherwise splits on newlines into <=_LINES_PER_CHUNK windows, capping at
    _MAX_CHUNKS (a very long film is sampled in _MAX_CHUNKS evenly-cut windows, not 1/window).
    """
    lines = text.split("\n")
    if len(lines) <= _CHUNK_LINES:
        return [text]
    # how many windows we'd need at _LINES_PER_CHUNK, but never exceed _MAX_CHUNKS
    n = min(_MAX_CHUNKS, max(2, -(-len(lines) // _LINES_PER_CHUNK)))   # ceil division
    size = -(-len(lines) // n)                                        # even window size
    return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)]


def _content_lemmas(text: str) -> set:
    """Lemmatized, stopword-filtered content tokens of `text` — the shared basis for the
    term↔surface relatedness check (S15 T1). Uses the same stopword set as the timestamp
    anchor (_common.stopwords_set) so grounding and provenance stay consistent."""
    from _common import stopwords_set
    stop = stopwords_set()
    out = set()
    for tok in _norm_match(text).split():
        if tok in stop:
            continue
        out.add(lemmatize_term(tok))
    return out


def _grounded(cand: dict, norm_text: str) -> bool:
    """True if the candidate is consistently grounded: its CITED SENTENCE really appears
    in the transcript AND the surface/term appears in THAT sentence.

    Stricter than a whole-transcript substring (which lets the AI pair a real term with the
    WRONG sentence — a provenance mismatch). Checking sentence-level catches a fabricated
    sentence (not in transcript) and a term-sentence mismatch in one go.

    S15 T1: when the match is via `surface` (not the `term` itself), the term and surface
    must share ≥1 CONTENT lemma. This blocks the "AI translated a foreign-language line into
    an English idiom" bug — e.g. term `do someone a favor` + surface "jouez-moi" both appear
    in a French line but share no lemma, so it is ungrounded (→ flagged → fix call).
    """
    sent = _norm_match(cand.get("sentence", ""))
    if not sent or sent not in norm_text:          # cited line must be a real transcript line
        return False
    term = cand.get("term", "")
    term_norm = _norm_match(term)
    # 1) the term itself appears verbatim in the line -> self-grounded, accept.
    if term_norm and term_norm in sent:
        return True
    # 2) matched via surface -> surface must be in the line AND share a content lemma
    #    with the term (else the AI paired an English term with an unrelated foreign word).
    surface_norm = _norm_match(cand.get("surface", ""))
    if surface_norm and surface_norm in sent:
        return bool(_content_lemmas(term) & _content_lemmas(cand.get("surface", "")))
    return False


def _parse_json_array(raw: str) -> list:
    """Strip an optional ```fence```, parse, and unwrap {"items":[...]} shapes."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


def _valid_items(parsed, seen: set, max_terms: int) -> list[dict]:
    """Validate + dedup a parsed array into Candidate dicts (clip-error: skip bad)."""
    out = []
    for item in parsed or []:
        try:
            cand = Candidate(**item) if isinstance(item, dict) else None
            if cand is None or not cand.term.strip() or not cand.sentence.strip():
                continue
            d = cand.model_dump()
            # Keep the ORIGINAL form (for grounding / provenance / .ass highlight), then
            # canonicalize the term to its lemma so inflected variants collapse to ONE node
            # (key = lemma#sense). dedup is BY LEMMA, so "emissions"/"emission" and
            # "figured out"/"figure out" no longer create duplicate nodes in a session.
            original = d["term"].strip()
            if not (d.get("surface") or "").strip():
                d["surface"] = original
            d["term"] = lemmatize_term(original)
            key = d["term"].strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(d)
            if len(out) >= max_terms:
                break
        except ValidationError:
            continue  # skip one bad item, keep going (clip-error)
    return out


def _gather_prompt(text: str, focus: str, max_terms: int, exclude: list[str]) -> str:
    parts = [f"User request / focus (follow it — topic and/or item types): {focus}\n" if focus else ""]
    parts.append(f"Return at most {max_terms} items.")
    if exclude:
        parts.append(
            "Do NOT return any of these already-selected terms (find DIFFERENT, "
            "additional useful items instead): " + ", ".join(exclude) + "."
        )
    parts.append(f"\nTRANSCRIPT:\n{text}")
    return "\n".join(p for p in parts if p)


def _fix_prompt(text: str, focus: str, flagged: list[str], keep: list[str]) -> str:
    return (
        (f"User request / focus (follow it — topic and/or item types): {focus}\n\n" if focus else "")
        + "Some previously selected terms do NOT actually appear in the transcript "
        "and must be replaced. For EACH listed bad term, return a DIFFERENT learning "
        "item that LITERALLY appears in the transcript below (set its real sentence + "
        "surface form). Do not reuse the already-kept terms.\n"
        f"Bad terms to replace: {', '.join(flagged)}\n"
        f"Already kept (do not repeat): {', '.join(keep) if keep else '(none)'}\n\n"
        f"TRANSCRIPT:\n{text}"
    )


def _call_array(user_prompt: str, *, mandatory: bool, args: dict, model: str | None = None) -> list:
    """One AI call → parsed JSON array. mandatory=True keeps the legacy parse-retry +
    SystemError_ contract (round 1). mandatory=False is best-effort (returns []).

    `model` routes this single call to a specific model (Day-5); None = config.AI_MODEL.
    """
    attempts = 2 if mandatory else 1
    for attempt in range(attempts):
        try:
            raw = call_ai(user_prompt, _SYSTEM_PROMPT, model)
            return _parse_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            if mandatory and attempt == attempts - 1:
                msg = f"extract_vocab: AI returned unparseable JSON twice: {e}"
                log_tool_call("extract_vocab", args, error=msg)
                raise SystemError_(msg)
            if not mandatory:
                return []
        except Exception as e:  # provider/network error from call_ai
            if mandatory:
                msg = f"extract_vocab: AI call failed: {e}"
                log_tool_call("extract_vocab", args, error=msg)
                raise SystemError_(msg)
            return []
    return []


def extract_vocab(transcript, focus: str = "", max_terms: int = 20) -> list[dict]:
    """Return a list of validated, transcript-grounded {term, sentence, surface, tag}.

    Bounded multi-call gather + Python self-correct + one fix call (see module doc).
    Deduplicated by lowercased term. Raises SystemError_ on missing key / repeated
    round-1 parse failure; skips individual malformed/ungrounded items (clip-error).
    """
    args = {"focus": focus, "max_terms": max_terms}
    if not config.has_ai_key():
        msg = ("No AI API key configured. Set GEMINI_API_KEY (or AI_API_KEY) in .env "
               "before calling extract_vocab.")
        log_tool_call("extract_vocab", args, error=msg)
        raise SystemError_(msg)

    text = _as_text(transcript).strip()
    if not text:
        log_tool_call("extract_vocab", args, result={"candidates": 0})
        return []

    norm_text = _norm_match(text)
    seen: set[str] = set()
    candidates: list[dict] = []
    calls_used = 0

    chunks = _split_into_chunks(text)
    if len(chunks) > 1:
        # --- 1'. LONG script: gather from EACH window for even coverage (HANDOVER §3.1) --- #
        # One AI call per window, excluding terms already found so windows surface DIFFERENT
        # items; the per-window target spreads the budget across the whole transcript instead
        # of front-loading it on the salient/early lines. Round-1 stays mandatory (key/parse
        # contract). Bounded by _MAX_CHUNKS (set when chunks were cut).
        # S12 T1: the effective cap SCALES with window count (min(_HARD_MAX, _PER_CHUNK*N)),
        # floored at _MIN_UNIQUE, so a long film is not short-changed by the default 20.
        cap = max(_MIN_UNIQUE, min(_HARD_MAX, _PER_CHUNK * len(chunks)))
        per_chunk = _PER_CHUNK
        for ci, chunk in enumerate(chunks):
            if len(candidates) >= cap:
                break
            exclude = [c["term"] for c in candidates]
            target = min(per_chunk, cap - len(candidates))
            prompt = _gather_prompt(chunk, focus, target, exclude)
            parsed = _call_array(prompt, mandatory=(ci == 0), args=args)
            calls_used += 1
            candidates.extend(_valid_items(parsed, seen, cap - len(candidates)))
    else:
        # --- 1. short text: bounded gather pass(es), round 1 mandatory (UNCHANGED) ------- #
        # Single-chunk demo clips keep the original 8-20 behaviour: cap == caller's max_terms.
        cap = max_terms
        gather_budget = max(1, _MAX_CALLS - 1)   # reserve at least one call for the fix pass
        while calls_used < gather_budget and len(candidates) < cap:
            mandatory = calls_used == 0
            exclude = [c["term"] for c in candidates]
            prompt = _gather_prompt(text, focus, cap, exclude)
            parsed = _call_array(prompt, mandatory=mandatory, args=args)
            calls_used += 1
            before = len(candidates)
            candidates.extend(_valid_items(parsed, seen, cap - len(candidates)))
            # S17 FIX: keep gathering toward `cap` (max_terms, default 20), NOT toward
            # _MIN_UNIQUE (8). The old `>= _MIN_UNIQUE` stop made a first call of ~10 items
            # end the loop, so a 500-line script yielded far fewer than it should. _MIN_UNIQUE
            # is a FLOOR, not the target. Stop only when we hit the cap or a pass adds nothing
            # (still bounded by gather_budget = _MAX_CALLS-1, so no runaway cost).
            if len(candidates) >= cap or len(candidates) == before:
                break

    # --- 2. Python self-correct: split grounded vs flagged (ungrounded) -------- #
    good = [c for c in candidates if _grounded(c, norm_text)]
    flagged = [c for c in candidates if not _grounded(c, norm_text)]

    # --- 3. one fix call: try to replace flagged terms with grounded ones ------ #
    # In chunked mode the gather already spent several calls; allow the fix call as long as we
    # are within the chunk budget (chunks + 1) rather than the short-path _MAX_CALLS cap.
    max_calls = _MAX_CALLS if len(chunks) == 1 else len(chunks) + 1
    if flagged and calls_used < max_calls:
        keep = [c["term"] for c in good]
        prompt = _fix_prompt(text, focus, [c["term"] for c in flagged], keep)
        # Route the self-correct fix call to the STRONG model (better grounding).
        # Free-tier gemini-2.5-flash is ~20/day, but this call only fires when the
        # gather pass produced ungrounded terms — typically ≤1 call per Mine run.
        fix_model = getattr(config, "AI_MODEL_STRONG", None)
        parsed = _call_array(prompt, mandatory=False, args=args, model=fix_model)
        calls_used += 1
        seen = {c["term"].strip().lower() for c in good}
        for c in _valid_items(parsed, seen, cap - len(good)):
            if _grounded(c, norm_text):       # only accept replacements that truly appear
                good.append(c)

    # --- 4. optional deterministic TYPE filter (hard guarantee, no extra AI call) --- #
    # If the focus named specific item types, drop any candidate whose tag is not one of
    # them — the prompt already biased toward them, so this rarely removes much.
    want = _requested_types(focus)
    if want:
        good = [c for c in good if _tag_norm(c.get("tag", "")) in want]

    result = good[:cap]
    log_tool_call("extract_vocab", args,
                  result={"candidates": len(result), "flagged": len(flagged),
                          "ai_calls": calls_used, "cap": cap, "type_filter": sorted(want)})
    return result


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else ""
    from ingest_transcript import ingest_transcript
    tr = ingest_transcript(src) if src else ""
    print(json.dumps(extract_vocab(tr), ensure_ascii=False, indent=2))
