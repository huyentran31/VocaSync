"""
_common.py — shared plumbing for the 10 tools.

Centralizes the things every tool needs so each tool file stays focused on its
one job and the project-wide rules (AGENTS.md) are enforced in ONE place:

  • paths            — project dir is the ONLY writable scope (least-privilege, Day-4)
  • graph load/save  — the PersonalGraph that grows across sessions (schema.py)
  • error tiers      — ClipError (mark fail + continue) vs SystemError (halt)
  • trajectory log   — one JSON line per tool call (observability / evals, Day-1/4)
  • secret masking   — never let an API key reach a log (context hygiene, Day-4)

Importing this also puts `legacy/` on sys.path so tools can reuse the bundled
modules (whisper_utils, ffmpeg, ai_client, file_utils, ...) and their `config` shim.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

# --- make repo root + legacy/ importable (legacy modules do `import config`) --- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(PROJECT_ROOT, "legacy")
for _p in (PROJECT_ROOT, LEGACY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema import PersonalGraph  # noqa: E402  (after sys.path setup)


# --------------------------------------------------------------------------- #
# Canonical paths (everything writable lives UNDER the project dir)
# --------------------------------------------------------------------------- #

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
GRAPH_PATH = os.path.join(DATA_DIR, "personal_graph.json")
TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "trajectory.jsonl")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_dir(run_id: str) -> str:
    """Per-run output folder: output/<run_id>/ (AGENTS.md §6)."""
    d = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Transcript cache (S17 ① — sentence∈transcript grounding)
# --------------------------------------------------------------------------- #
# When a transcript is ingested, its full text is stashed keyed by the source's
# basename. stage_for_review reads it back to verify that a sentence the agent
# cites for a kept word is a REAL transcript line (not fabricated) — the second
# grounding tier on the sole agent write path. Basename is the shared key because
# ingest gets the full path but stage_for_review reduces `source` to its basename.
# Under output/ (gitignored), inside the project dir (least-privilege). ADDITIVE.

TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "_transcripts")


def _transcript_cache_path(source: str) -> str:
    return os.path.join(TRANSCRIPTS_DIR, ascii_safe(os.path.basename(source)) + ".txt")


def _srtpath_cache_path(source: str) -> str:
    """Sidecar next to the transcript text cache, holding the SRT/segments file path (S18 #2)."""
    return os.path.join(TRANSCRIPTS_DIR, ascii_safe(os.path.basename(source)) + ".srtpath")


def cache_transcript(source: str, full_text: str, srt_path: str = "") -> None:
    """Stash a transcript's full text keyed by source basename (no-crash).

    S18 #2 (ADDITIVE): also stash the timestamped `srt_path` in a sidecar so the commit step
    can recover clip timings for AGENT-staged words when the sidebar's `last_srt` is empty
    (mine-via-chat: the agent ingested the media, but that srt path never reached commit)."""
    try:
        if not source or not full_text:
            return
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        with open(_transcript_cache_path(source), "w", encoding="utf-8") as f:
            f.write(full_text)
        if srt_path:
            with open(_srtpath_cache_path(source), "w", encoding="utf-8") as f:
                f.write(srt_path)
    except Exception:
        pass  # caching must never break ingestion


def load_cached_transcript(source: str) -> str:
    """Return the cached transcript text for `source` (basename), or "" if none."""
    try:
        p = _transcript_cache_path(source)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""


def load_cached_srt(source: str) -> str:
    """Return the cached SRT/segments path for `source` (basename), or "" if none/missing (S18 #2)."""
    try:
        p = _srtpath_cache_path(source)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                sp = f.read().strip()
            if sp and os.path.exists(sp):
                return sp
    except Exception:
        pass
    return ""


def new_run_id() -> str:
    """Timestamped, filesystem-safe run id. Caller-supplied time (never hidden)."""
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    """ISO date string — callers pass this into Occurrence.added_at, etc."""
    return datetime.now(timezone.utc).date().isoformat()


def in_project(path: str) -> bool:
    """Guardrail: is `path` inside the project dir? (deny writes outside.)"""
    try:
        return os.path.commonpath([os.path.abspath(path), PROJECT_ROOT]) == PROJECT_ROOT
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Error tiers (AGENTS.md §4 / error_handling.gherkin)
# --------------------------------------------------------------------------- #

class ClipError(Exception):
    """One item failed (bad word / timestamp / single clip). Mark fail + CONTINUE."""


class SystemError_(Exception):
    """System-level fault (missing key / corrupt graph / ffmpeg|whisper missing). HALT."""


# --------------------------------------------------------------------------- #
# PersonalGraph load/save (the memory that grows — AGENTS.md §5)
# --------------------------------------------------------------------------- #

def load_graph(path: str = GRAPH_PATH) -> PersonalGraph:
    """Load the graph; a corrupt file is a SYSTEM error (never silently reset)."""
    try:
        return PersonalGraph.load(path)
    except Exception as e:
        raise SystemError_(
            f"PersonalGraph at {path} is corrupt or unreadable: {e}. "
            f"Fix/restore the file before continuing (refusing to overwrite)."
        )


def save_graph(graph: PersonalGraph, path: str = GRAPH_PATH) -> None:
    if not in_project(path):
        raise SystemError_(f"Refusing to write graph outside project dir: {path}")
    ensure_dirs()
    graph.save(path)


# --------------------------------------------------------------------------- #
# Secret masking + trajectory log (Day-4 context hygiene / observability)
# --------------------------------------------------------------------------- #

_SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)")
# `key=AIza...` / `token: xxxx` — redact the VALUE, not just the keyword
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|key|token|secret|authorization|bearer)(\s*[=:]\s*)([A-Za-z0-9_\-\.]{12,})")
# any Google-API-key-shaped string, anywhere
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def mask_secrets(obj):
    """Recursively redact anything that looks like a secret before logging."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_RE.search(str(k)):
                out[k] = "***"
            else:
                out[k] = mask_secrets(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [mask_secrets(x) for x in obj]
    if isinstance(obj, str):
        # redact secret VALUES (key=..., token: ...) and any Google-key-shaped string
        s = _SECRET_VALUE_RE.sub(lambda m: m.group(1) + m.group(2) + "***", obj)
        s = _GOOGLE_KEY_RE.sub("***", s)
        return s
    return obj


def _summarize(result) -> object:
    """Keep the trajectory log small: summarize big results, don't dump them."""
    try:
        if isinstance(result, list):
            return {"type": "list", "len": len(result)}
        if isinstance(result, dict):
            return {"type": "dict", "keys": list(result.keys())[:12]}
        s = str(result)
        return s if len(s) <= 300 else s[:300] + "…"
    except Exception:
        return "<unserializable>"


def log_tool_call(tool: str, args: dict, result=None, error: str = "",
                  path: str = TRAJECTORY_PATH) -> None:
    """Append one JSON line: {ts, tool, args, result|error}. Never raises."""
    try:
        ensure_dirs()
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "args": mask_secrets(args or {}),
        }
        if error:
            entry["error"] = error
        else:
            entry["result"] = _summarize(result)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break the run


def ascii_safe(name: str) -> str:
    """Make a filename ASCII-only (AGENTS.md §6 — Anki media must be ASCII)."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    return name.strip("_") or "x"


# --------------------------------------------------------------------------- #
# Stopwords (shared: agent/loop timestamp anchor + extract_vocab grounding)
# --------------------------------------------------------------------------- #

# Fallback set when the NLTK corpus is missing (no-crash) — the common function words
# that must NOT be used as a content anchor (they match almost any line).
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "to", "from", "in", "on", "off", "out", "up", "down", "over", "under",
    "be", "am", "is", "are", "was", "were", "been", "being", "do", "does", "did",
    "have", "has", "had", "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "my", "your", "his", "her", "their", "our", "this", "that", "as", "so", "not", "no",
}


def stopwords_set() -> set:
    """English stopwords via NLTK if available, else the hardcoded fallback (no-crash)."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


# --------------------------------------------------------------------------- #
# Shared grounding (S18 HEART) — Python OWNS the quoted line; AI never writes it
# --------------------------------------------------------------------------- #
# ONE matcher for BOTH write paths that show the learner a transcript line:
#   • Ask  — explain's "From this video" block (agent/loop._materials_for)
#   • Mine — the source sentence that decides the Anki AUDIO cut (stage_for_review)
# Each must resolve a term to a VERBATIM cue or to nothing (never an AI-authored
# line). Matching is strict-first so an inflected variant is caught, but a bare
# substring ("turn" in "returns") or a scrambled run is NOT. This replaces the two
# divergent matchers that each had the same word-boundary hole (LEAD 2026-07-04).
# Lesson from `AI Teaching 7 Jun 1`: check ≡ show — grounding uses the SAME
# normalization the learner-facing text is quoted with.

def _ground_norm(s: str) -> str:
    """Normalization shared by grounding AND display: lowercase, keep [a-z0-9']
    tokens, single-spaced. (Apostrophe kept so "don't" stays one token.)"""
    return " ".join(re.findall(r"[a-z0-9']+", str(s).lower()))


# Reflexive pronouns collapse to "oneself": a dictionary-form phrasal like "knock ONESELF out"
# must ground its real cue "knock YOURSELF out." (askfix V4 — the miss kept an AI-invented
# sentence on the card, so the audio was cut from the wrong line).
_REFLEXIVES = {"myself", "yourself", "yourselves", "himself", "herself", "itself",
               "ourselves", "themselves", "oneself"}


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=20000)
def _lemma_seq(s: str) -> tuple:
    """Ordered lemma tuple of `s` — NOT stopword-filtered: order and function words
    matter here ("turn to" -> (turn, to) must stay distinct from "turn too" ->
    (turn, too)). Reflexive pronouns normalize to "oneself" (dictionary form ↔ speech).
    Degrades to the raw normalized tokens if WordNet is unavailable.

    askfix (owner V2.1 speed): memoized + returns a tuple — grounding calls this on the SAME
    SRT line once per candidate; caching makes repeated calls instant. Read-only for all callers."""
    try:
        from extract_vocab import lemmatize_term
    except Exception:
        return tuple("oneself" if t in _REFLEXIVES else t for t in _ground_norm(s).split())
    return tuple("oneself" if t in _REFLEXIVES else lemmatize_term(t)
                 for t in _ground_norm(s).split())


def _contains_run(hay: list, needle: list) -> bool:
    """True iff `needle` occurs as a CONTIGUOUS, in-order sublist of `hay`."""
    n, m = len(hay), len(needle)
    if not m or m > n:
        return False
    return any(hay[i:i + m] == needle for i in range(n - m + 1))


# Separable phrasal-verb support (S18 askfix): a particle verb can take its object BETWEEN
# the verb and the particle ("put her down", "turn it off"). We match "put down" against
# "put her down" ONLY when the inserted tokens are PRONOUNS — the common separable case in
# speech — so we never falsely match a compound like "put the down payment" (gap "the" is a
# determiner, rejected) or two unrelated words far apart. Deliberately conservative: a
# determiner+noun object ("put the baby down") is NOT matched (misses some, guesses none).
_PARTICLES = {"up", "down", "off", "on", "out", "in", "away", "back", "over",
              "around", "round", "through", "along", "apart", "aside", "by"}
_PRONOUN_OBJECTS = {"it", "them", "him", "her", "me", "us", "you", "one", "'em",
                    "oneself", "myself", "yourself", "himself", "herself", "itself",
                    "ourselves", "themselves", "this", "that"}


def _phrasal_split_match(term_lemmas: list, line_lemmas: list) -> bool:
    """True iff `term_lemmas` is a 2-word [verb, particle] that appears in `line_lemmas` with
    the verb and particle separated only by PRONOUN object(s) (gap of 1-2 pronouns)."""
    if len(term_lemmas) != 2 or term_lemmas[1] not in _PARTICLES:
        return False
    v, p = term_lemmas
    for i, tok in enumerate(line_lemmas):
        if tok != v:
            continue
        for j in range(i + 2, min(len(line_lemmas), i + 4)):   # gap of 1-2 tokens
            if line_lemmas[j] == p and all(
                    line_lemmas[k] in _PRONOUN_OBJECTS for k in range(i + 1, j)):
                return True
    return False


def _defl(t: str) -> str:
    """Fold an -ing/-in' form to its bare stem so a lemma token ("going") can match ("go");
    -ed/-s are already handled by lemmatize_term (see _lemma_seq). Conservative — only -ing/-in',
    only on longish tokens, so it never collapses short unrelated words."""
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("in'"):
        return t[:-3]
    return t


def _infl_eq(a: str, b: str) -> bool:
    """Token equality tolerant of an -ing verb inflection ("going" == "go", "eating" == "eat")."""
    return a == b or _defl(a) == _defl(b)


def _contains_run_infl(hay: list, needle: list) -> bool:
    """_contains_run, but element comparison is inflection-tolerant (_infl_eq). Still CONTIGUOUS
    and in-order, so it adds no gap risk — only tolerant token equality."""
    n, m = len(hay), len(needle)
    if not m or m > n:
        return False
    return any(all(_infl_eq(hay[i + k], needle[k]) for k in range(m))
               for i in range(n - m + 1))


# Grammatical fillers that may sit BETWEEN a multi-word term's tokens in real speech without
# changing the phrase ("take THIS the wrong way", "eating ME out of house and home"). Pronoun
# objects + articles + demonstratives + possessives. Used ONLY by the >=3-token gap tier below,
# so 2-word phrasals keep TIER 3's stricter pronoun-only rule (never match "put THE down payment").
_GAP_FILLERS = _PRONOUN_OBJECTS | {"the", "a", "an", "this", "that", "these", "those",
                                   "my", "your", "his", "her", "its", "our", "their", "'s"}


def _gap_ordered_infl(term_lemmas: list, line_lemmas: list) -> bool:
    """True iff every term token appears IN ORDER in the line (inflection-tolerant), where any
    gap between two consecutive term tokens is only 1-2 grammatical FILLERS. Multi-word terms
    only (the caller restricts to >=3 tokens). Conservative: a content-word gap breaks the match."""
    n = len(line_lemmas)
    for start in range(n):
        if not _infl_eq(line_lemmas[start], term_lemmas[0]):
            continue
        li, ok = start, True
        for ti in range(1, len(term_lemmas)):
            found = False
            for j in range(li + 1, min(n, li + 4)):        # gap of 0-2 tokens between
                if _infl_eq(line_lemmas[j], term_lemmas[ti]) and all(
                        line_lemmas[k] in _GAP_FILLERS for k in range(li + 1, j)):
                    li, found = j, True
                    break
            if not found:
                ok = False
                break
        if ok:
            return True
    return False


def ground_line(term: str, lines, ai_hint: str = ""):
    """Resolve `term` to the VERBATIM transcript line it belongs to, or None.

    `lines` is a list of candidate cue strings (SRT segments for Mine, sentence-split
    transcript lines for Ask). Python OWNS the returned line — it is always a real
    cue, never text an AI wrote. Two tiers, strictest first:

      1) word-bounded verbatim phrase — " term " inside " line " (normalized). A
         repeated phrase yields several cues; `ai_hint` (an AI-cited sentence) picks
         the closest real one, else the FIRST (still verbatim, so still a real line).
      2) ordered lemma-run — the term's lemmas appear as a CONTIGUOUS, in-order run
         in a line's lemmas. Catches inflection ("turn to" ↔ "turned to") while
         rejecting a bare substring ("returns" -> [return] ≠ [turn]) and a scramble
         ("turn too fat"). Ambiguous (several cues) with NO hint -> None, so the
         caller flags/drops rather than guessing the clip from the wrong line.

    Returns None when nothing matches (caller: "Not found in this video." / gate).
    """
    lns = [str(ln).strip() for ln in (lines or []) if str(ln).strip()]
    if not term or not lns:
        return None
    tn = _ground_norm(term)
    hint = _ground_norm(ai_hint) if ai_hint else ""

    def _pick(cands):
        """Disambiguate several real cues: an AI hint chooses the closest; no hint -> None."""
        if len(cands) == 1:
            return cands[0]
        if hint:
            exact = [c for c in cands if _ground_norm(c) == hint]
            if exact:
                return exact[0]                # AI cited one of the real cues verbatim
            toks = set(hint.split())
            return max(cands, key=lambda c: len(toks & set(_ground_norm(c).split())) / (len(toks) or 1))
        return None

    # TIER 1 — verbatim phrase with word boundaries.
    verbatim = [ln for ln in lns if tn and f" {tn} " in f" {_ground_norm(ln)} "]
    if verbatim:
        return _pick(verbatim) or verbatim[0]   # every hit is a real line -> first is safe

    # TIER 2 — ordered lemma-run (inflected variants); ambiguous + no hint -> None.
    tl = _lemma_seq(term)
    run = [ln for ln in lns if _contains_run(_lemma_seq(ln), tl)]
    if run:
        return _pick(run)

    # TIER 3 — separable phrasal verb: "put down" grounds "...put her down." (pronoun object
    # inserted). Conservative (pronoun gap only), so it never invents a match. This is the fix
    # for the mined "put down" card whose sentence used to stay the AI's fabricated line.
    split = [ln for ln in lns if _phrasal_split_match(tl, _lemma_seq(ln))]
    if split:
        return _pick(split) or split[0]

    # TIER 4 (S19) — inflection-tolerant CONTIGUOUS run: catches "go solo" ↔ "going solo" (only a
    # verb -ing form differs; -ed/-s are already folded upstream). Same contiguity as TIER 2, so
    # no new gap risk — just tolerant token equality. Ambiguous + no hint -> fall through.
    if len(tl) >= 2:
        infl = [ln for ln in lns if _contains_run_infl(_lemma_seq(ln), tl)]
        if infl and (res := _pick(infl)):
            return res

    # TIER 5 (S19) — function-word-gap ordered match for MULTI-word terms (>=3 tokens): the term's
    # tokens appear in order with only short grammatical fillers inserted between them. Catches
    # "take the wrong way" ↔ "take THIS the wrong way" and "eat out of house and home" ↔ "eatING ME
    # out of house and home". Restricted to >=3 tokens so 2-word phrasals keep TIER 3's PRONOUN-only
    # rule (this never matches "put THE down payment"). A content-word gap breaks the match.
    if len(tl) >= 3:
        gap = [ln for ln in lns if _gap_ordered_infl(tl, _lemma_seq(ln))]
        if gap and (res := _pick(gap)):
            return res

    # TIER 6 (S19 OPEN-6i) — dictionary "be" prefix: the extractor emits copula-headed idioms
    # ("be all over the place", "be the happs") but the film says "They're all over the place" /
    # "what's the happs?" (no "be"). Retry the WHOLE tier stack on the remainder. Guards, so it
    # never loosens an existing match: runs ONLY after every tier above failed, strips ONLY a
    # leading "be" (owner: have/get/do carry real meaning too often to drop), and requires the
    # remainder to keep >=2 tokens (so a bare "be up" -> "up" can't match any line loosely).
    toks = tn.split()
    if len(toks) >= 3 and toks[0] == "be":
        return ground_line(" ".join(toks[1:]), lns, ai_hint)
    return None
