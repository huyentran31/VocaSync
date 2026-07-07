"""
enrich.py — Tool #5 (write-draft, EXPENSIVE, AI). The deterministic-first keystone.

ONE AI call for a whole batch (execution_policy: enrich.batch=true). For each term
the model does exactly two things:
  1. PICK the correct WordNet sense — but ONLY from the sense_ids we hand it
     (constrained choice → it cannot invent a sense).
  2. FILL uncertain fields (collocations, mnemonic, pattern) — these are flagged
     source='ai' in source_map and must pass HITL review.

Everything grounded (sense's edges, category, pos) is copied DETERMINISTICALLY from
the chosen WordNet sense — never from the model (enrich.gherkin). If a term has no
WordNet sense, the model may supply an AI definition (flagged) and the unit is forced
to HITL review.

Output: list of draft units {node, confidence, needs_review, ai_fields}. The Node is
validated against schema.py; build_render_graph / make_anki consume node, the in-app review table
consumes the rest. Nothing is committed to the graph here (HITL gate, AGENTS.md §5).
"""

from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

import config
from ai_client import call_ai
from _common import SystemError_, log_tool_call, now_iso
from schema import Node, Edge, Occurrence, Media, normalize_collocations

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "description": "Items to enrich; each {term, sentence, senses, surface?, source?}.",
        },
        "source": {"type": "string", "description": "Default occurrence source label."},
        "focus": {"type": "string", "description": "Optional topic context for disambiguation."},
    },
    "required": ["units"],
}

# fields the model is allowed to author (everything else is WordNet/deterministic)
_AI_FIELDS = ("sense_id", "collocations", "mnemonic", "pattern")
# knobs read from .env (execution_policy.yaml documents them; config.py never defined them,
# so the old config.__dict__ lookup always fell back to the default — S14 T6)
_LOW_CONF = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.7"))
# heavy-polysemy flag (S14 T7): a term with >= this many candidate senses is where sense
# disambiguation actually fails (eval: "run", "light") — force review regardless of the
# model's self-reported confidence. Deterministic, no AI.
_POLYSEMY_REVIEW_MIN = int(os.getenv("POLYSEMY_REVIEW_MIN", "5"))

_SYSTEM_PROMPT = (
    "You are a careful lexicographer assistant for an English-learning tool. "
    "For each term you are given the sentence it appeared in and a numbered list of "
    "candidate WordNet senses (id + definition). Do FOUR things per term:\n"
    "1) Choose the sense that best fits the sentence — you MUST return one of the "
    "given sense_id strings exactly (or null only if no candidate fits).\n"
    "1b) Return `usage_pos` = the grammatical part of speech of the term AS USED in the "
    "sentence. Decide this from the SENTENCE SYNTAX FIRST, independently of the candidate "
    "senses listed. One of: noun | verb | adj | adv (or null if truly unclear).\n"
    "2) Provide up to 3 natural collocations, one short memorable mnemonic, and one "
    "usage pattern.\n"
    "3) If a term lists 'Candidate context edges' (from ConceptNet, which is "
    "sense-agnostic), return `keep_edges`: the list of those edge TARGET strings that "
    "are consistent with the sense you picked. Drop ones that belong to a DIFFERENT "
    "meaning (e.g. for 'spring' as a season, drop 'jump'). If the term has no candidate "
    "edges, return an empty list. Do NOT add edges that were not offered.\n"
    "4) Return `tags`: 1-3 short topic/domain labels for this term, ordered "
    "most-relevant first (e.g. [\"finance\",\"business\"]). If a Focus topic is given, "
    "include it when appropriate. Lowercase, single words/short phrases.\n"
    "\n"
    "CONFIDENCE (0..1) for your sense choice — assign by this rubric [FIRST DRAFT, refine later]:\n"
    "  0.9-1.0 : the sentence unambiguously matches exactly one sense; no plausible alternative.\n"
    "  0.7-0.85: the best sense is clear but another sense is somewhat plausible.\n"
    "  0.5-0.65: two or more senses are plausible; the choice is a judgment call.\n"
    "  <0.5    : guessing, OR the usage's part-of-speech does NOT match the sense's POS\n"
    "            (e.g. a verb usage forced onto a noun sense) — cap at 0.5 in that case.\n"
    "  If NO candidate sense fits, return sense_id=null with confidence <= 0.3.\n"
    "\n"
    "Do NOT invent synonyms, antonyms or definitions when WordNet senses ARE offered — "
    "those come from WordNet/ConceptNet. EXCEPTION: if a term has NO candidate senses "
    "(none in WordNet — an idiom/collocation), return a short plain-English `definition` "
    "for it (this is the ONLY case you may author a definition; leave it out otherwise). "
    "Return STRICT JSON: an array of "
    '{"term","sense_id","usage_pos","definition","collocations":[...],"mnemonic","pattern",'
    '"confidence","tags":[...],"keep_edges":[...]}. '
    "Output ONLY the JSON array."
)


def _lemmas(sentence: str) -> list[str]:
    """Precompute lemmas for cheap recall. Use spaCy if available, else lowercased tokens."""
    try:
        import spacy  # optional, heavy — graceful fallback if absent
        nlp = _spacy_model()
        return [t.lemma_.lower() for t in nlp(sentence) if t.is_alpha]
    except Exception:
        return re.findall(r"[a-z]+", sentence.lower())


_NLP = None
def _spacy_model():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _parse_json_array(raw: str) -> list:
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
    return parsed


def _sense_by_id(senses: list[dict], sense_id) -> dict | None:
    for s in senses:
        if s.get("sense_id") == sense_id:
            return s
    return None


def _build_prompt(units: list[dict], focus: str) -> str:
    parts = [f"Focus topic: {focus}\n" if focus else ""]
    for i, u in enumerate(units, start=1):
        parts.append(f"=== TERM {i}: {u['term']} ===")
        parts.append(f'Sentence: "{u.get("sentence","")}"')
        senses = u.get("senses", []) or []
        if senses:
            parts.append("Candidate senses:")
            for s in senses[:8]:
                parts.append(f"  - {s['sense_id']}: {s.get('definition','')}")
        else:
            parts.append("Candidate senses: (none in WordNet — you may give an AI definition)")
        cn_edges = u.get("cn_edges", []) or []
        if cn_edges:
            parts.append("Candidate context edges (ConceptNet — keep only those fitting the chosen sense):")
            for e in cn_edges[:12]:
                parts.append(f"  - {e.get('type')} -> {e.get('target')}")
        parts.append("")
    return "\n".join(parts)


def _draft_for(unit: dict, ai: dict, default_source: str) -> dict:
    """Assemble ONE schema-valid Node from deterministic WordNet data + flagged AI fields."""
    term = unit["term"].strip()
    sentence = unit.get("sentence", "")
    surface = unit.get("surface", "") or term
    source = unit.get("source") or default_source
    senses = unit.get("senses", []) or []
    # word_type: persist the classification extract_vocab already made (Phrasal Verb / Idiom /
    # Collocation / Slang / Word) so idioms & phrasal verbs are visible/filterable downstream.
    # askfix (owner V5): the CHAT-staged path carries no `tag`, so every multi-word phrase fell
    # back to plain "word" in the graph/infolog. When the tag is missing, classify
    # deterministically from the term's own shape: verb+particle -> phrasal_verb, any other
    # multi-word -> collocation, single word -> word. Tag (when present) still wins.
    _tag = (unit.get("tag", "") or "").strip().lower()
    if not _tag:
        _toks = str(term).lower().split()
        if len(_toks) >= 2:
            try:
                from _common import _PARTICLES
                _tag = "phrasal_verb" if _toks[-1] in _PARTICLES else "collocation"
            except Exception:
                _tag = "collocation"
        else:
            _tag = "word"
    word_type = re.sub(r"[^a-z]+", "_", _tag).strip("_") or "word"

    ai = ai or {}
    chosen_id = ai.get("sense_id")
    confidence = float(ai.get("confidence", 0.0) or 0.0)
    needs_review = False
    ai_flagged = []
    source_map: dict[str, str] = {}

    sense = _sense_by_id(senses, chosen_id)
    if sense is None and senses:
        # AI gave an out-of-list / null sense → fall back to most common (sense[0]),
        # lower confidence, force review.
        sense = senses[0]
        chosen_id = sense["sense_id"]
        confidence = min(confidence, 0.5)
        needs_review = True

    edges, category, pos, sense_id, definition = [], None, None, None, None
    if sense is not None:
        sense_id = sense["sense_id"]
        category = sense.get("category")
        pos = sense.get("pos")
        definition = sense.get("definition")  # WordNet gloss — deterministic, the learner-facing meaning
        edges = [Edge(**e) for e in sense.get("edges", [])]   # deterministic, source="wordnet"
        source_map["sense_id"] = "ai"        # the CHOICE was AI's
        ai_flagged.append("sense_id")
        if category:
            source_map["category"] = "wordnet"
        if pos:
            source_map["pos"] = "wordnet"
        if definition:
            source_map["definition"] = "wordnet"
        if edges:
            source_map["edges"] = "wordnet"
        # Deterministic POS cross-check: the MODEL reported usage_pos from the sentence
        # syntax; the COMPARISON + cap live HERE in code (deterministic-first). A verb
        # used in a noun sense (e.g. "run" -> run.n.07) is the eval's known failure mode
        # -> cap confidence and force review. No-op if the model gave no usage_pos.
        usage_pos = (ai.get("usage_pos") or "").strip().lower()
        if usage_pos and pos and usage_pos != str(pos).strip().lower():
            confidence = min(confidence, 0.5)
            needs_review = True
            source_map["usage_pos"] = "ai"
    else:
        # not in WordNet → no sense to be confident ABOUT. Enforce the rubric's
        # "null/OOV -> confidence <= 0.3" in CODE (the model ignores it: it returns 1.0
        # for OOV terms). Everything here needs review.
        needs_review = True
        confidence = min(confidence, 0.3)
        # S15 T2: OOV terms (idioms / collocations WordNet doesn't know) may carry an
        # AI-authored definition — the prompt promises it. Flag it source='ai' so it is
        # visible/editable in review; without this the row reaches review with definition
        # None and the commit gate (validate_edits) blocks it until the human writes one.
        ai_def = (ai.get("definition") or "").strip()
        if ai_def:
            definition = ai_def
            source_map["definition"] = "ai"     # AI-authored, NOT WordNet
            ai_flagged.append("definition")

    # --- ConceptNet "life-context" edges (sense-agnostic) the AI vetted against the chosen sense ---
    # Candidates come deterministically from conceptnet_lookup (source="conceptnet"); the AI only
    # FILTERS them via `keep_edges` (it cannot add edges that were not offered). Anything kept is
    # flagged for HITL review. Missing keep_edges -> keep all candidates (HITL still gates them).
    cn_edges = unit.get("cn_edges", []) or []
    if cn_edges:
        keep = ai.get("keep_edges")
        if isinstance(keep, list):
            keepset = {str(k).strip().lower() for k in keep}
            cn_edges = [e for e in cn_edges if (e.get("target") or "").strip().lower() in keepset]
        seen_edges = {(e.type, e.target) for e in edges}
        for ce in cn_edges:
            try:
                edge = Edge(**ce) if not isinstance(ce, Edge) else ce
            except Exception:
                continue
            if (edge.type, edge.target) in seen_edges:
                continue
            seen_edges.add((edge.type, edge.target))
            edges.append(edge)              # Edge.source == "conceptnet" carries provenance per-edge
        if any(e.source == "conceptnet" for e in edges):
            needs_review = True             # ConceptNet is noisier than WordNet -> always review
            ai_flagged.append("conceptnet_edges")

    # AI uncertain fields (flagged)
    # S19 BUG-1: LLM sometimes returns collocations as a STRING; normalize_collocations
    # splits it into phrases instead of letting `for c in <str>` explode it per-character.
    collocations = normalize_collocations(ai.get("collocations"))
    mnemonic = (ai.get("mnemonic") or "").strip() or None
    pattern = (ai.get("pattern") or "").strip() or None
    # Topic/domain tags (AI-proposed, ordered most-relevant first; learner edits in review).
    tags, _seen_tags = [], set()
    for t in (ai.get("tags") or []):
        t = str(t).strip().lower()
        if t and t not in _seen_tags:
            _seen_tags.add(t)
            tags.append(t)
    tags = tags[:3]
    for f, val in (("collocations", collocations), ("mnemonic", mnemonic),
                   ("pattern", pattern), ("tags", tags)):
        if val:
            source_map[f] = "ai"
            ai_flagged.append(f)

    if confidence < _LOW_CONF:
        needs_review = True

    # Heavy polysemy -> always review (S14 T7): confidence is NOT lowered, only the flag set.
    if len(senses) >= _POLYSEMY_REVIEW_MIN:
        needs_review = True
        if "polysemy" not in ai_flagged:
            ai_flagged.append("polysemy")

    key = f"{term.lower()}#{sense_id}" if sense_id else f"{term.lower()}#nowordnet"
    occ = Occurrence(
        source=source, sentence=sentence, lemmas=_lemmas(sentence),
        surface=surface,                                        # the ORIGINAL form seen here
        media=Media(), added_at=now_iso(),
        start=unit.get("start", ""), end=unit.get("end", ""),   # provenance: where in the media
    )
    node = Node(
        key=key, term=term, word_type=word_type, sense_id=sense_id, pos=pos, category=category,
        definition=definition,
        edges=edges, collocations=collocations, mnemonic=mnemonic, pattern=pattern,
        tags=tags,
        occurrences=[occ], source_map=source_map,
    )
    return {
        "node": node.model_dump(),
        "confidence": confidence,
        "needs_review": needs_review,
        "ai_fields": ai_flagged,
        "surface": surface,
    }


def enrich(units: list[dict], source: str = "manual_import", focus: str = "") -> list[dict]:
    """Enrich a batch of units in ONE AI call. Returns draft units (see module doc).

    Raises SystemError_ on missing key. AI/parse failures degrade to a deterministic
    fallback (sense[0], no AI fields, needs_review=True) rather than crashing.
    """
    args = {"n_units": len(units or []), "source": source, "focus": focus}
    if not units:
        return []
    if not config.has_ai_key():
        msg = "No AI API key configured (set GEMINI_API_KEY/AI_API_KEY in .env)."
        log_tool_call("enrich", args, error=msg)
        raise SystemError_(msg)

    prompt = _build_prompt(units, focus)
    ai_by_term: dict[str, dict] = {}
    for attempt in range(2):  # one AI call, parse-retry once
        try:
            raw = call_ai(prompt, _SYSTEM_PROMPT)
            for item in _parse_json_array(raw) or []:
                if isinstance(item, dict) and item.get("term"):
                    ai_by_term[str(item["term"]).strip().lower()] = item
            break
        except (json.JSONDecodeError, ValueError):
            if attempt == 1:
                ai_by_term = {}  # give up on AI fields → deterministic fallback below
        except Exception as e:
            # provider/network error: don't crash the batch, fall back deterministically
            log_tool_call("enrich", args, error=f"AI call failed, using fallback: {e}")
            ai_by_term = {}
            break

    drafts = []
    for u in units:
        ai = ai_by_term.get(u["term"].strip().lower(), {})
        try:
            drafts.append(_draft_for(u, ai, source))
        except ValidationError as e:
            # schema build failed for this one item → retry once with no AI fields
            try:
                drafts.append(_draft_for(u, {}, source))
            except Exception:
                log_tool_call("enrich", args, error=f"skipped {u.get('term')}: {e}")
                continue

    n_review = sum(1 for d in drafts if d["needs_review"])
    log_tool_call("enrich", args, result={"drafts": len(drafts), "needs_review": n_review})
    return drafts


if __name__ == "__main__":
    import sys
    from wordnet_lookup import wordnet_lookup
    term = sys.argv[1] if len(sys.argv) > 1 else "gas"
    sent = sys.argv[2] if len(sys.argv) > 2 else "reduce carbon gas emissions"
    u = [{"term": term, "sentence": sent, "senses": wordnet_lookup(term)["senses"]}]
    print(json.dumps(enrich(u, source="demo"), ensure_ascii=False, indent=2))
