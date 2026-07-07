"""
stage_for_review.py — Tool #10 (write to the REVIEW QUEUE, never the graph).

The FIRST agent-callable WRITE tool. When the learner explicitly wants to keep a word
they are discussing, the agent stages it into the HITL review queue (pending_drafts.json,
reviewed in-app via st.data_editor) for later approval — it NEVER touches personal_graph.json.
That single graph-commit point stays in app.commit_approved (HITL contract, AGENTS.md §5).

It runs the same deterministic-first pipeline the Mine flow uses (wordnet_lookup -> enrich,
AI-flagged) for the given `terms`, or accepts already-enriched `drafts`, then appends them
to the queue via review_io.export_review(mode="append") (dedup by key; existing rows kept).

No-crash: a bad term is skipped and logged; the tool returns what it managed to stage.
"""

from __future__ import annotations

import re

from _common import log_tool_call

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {"type": "array", "items": {"type": "string"},
                  "description": "Words/phrases the learner asked to keep (base form)."},
        "sentences": {"type": "object",
                      "description": "term -> the EXACT source line the word appeared in. "
                                     "REQUIRED for grounding: a term with no source sentence "
                                     "(or a sentence that does not actually contain the word) "
                                     "is NOT staged and is returned in `ungrounded`."},
        "drafts": {"description": "Optional already-enriched drafts ({node,...}) to stage as-is."},
        "source": {"type": "string", "description": "Where the word came from (e.g. a film name)."},
    },
    "required": [],
}


def stage_for_review(terms=None, drafts=None, source: str = "", sentences=None) -> dict:
    """Stage words the learner wants to keep INTO the review queue (NOT the graph).

    Runs wordnet_lookup + enrich for `terms` (or accepts ready enrich `drafts`), then
    review_io.export_review(..., mode="append"). NEVER writes personal_graph.json.
    Returns {"staged": [all terms processed], "newly_staged": [terms added as NEW undecided
    rows], "already_present": [terms already in the queue — kept as-is, no new row],
    "review_path": ...}. The agent should report newly_staged vs already_present so the learner
    isn't told "saved N" when most were duplicates already sitting in the queue.
    """
    args = {"terms": terms, "source": source, "has_drafts": bool(drafts)}
    import os
    # Same source LABEL as Mine (which uses basename): a path-looking source is reduced to
    # its filename so occurrence dedup sees ONE label per film, not "X" vs "X.srt" (S14 T11).
    if source and (os.sep in source or "/" in source or "\\" in source):
        source = os.path.basename(source)
    from wordnet_lookup import wordnet_lookup
    from enrich import enrich
    import review_io

    # Accept messy agent input: `terms` may be a list or a single string; `drafts` may be
    # ready enrich drafts ({"node": {...key...}}) OR flat {"term": ...} dicts OR bare strings.
    # Anything that isn't a fully-formed draft is reduced to a term string and enriched here,
    # so a word the learner asked to keep is NEVER silently dropped.
    if isinstance(terms, str):
        terms = [terms]
    want_terms = [str(t).strip() for t in (terms or []) if str(t).strip()]
    staged_drafts = []
    for d in (drafts or []):
        if isinstance(d, dict) and isinstance(d.get("node"), dict) and d["node"].get("key"):
            staged_drafts.append(d)                 # already a proper enrich draft
        elif isinstance(d, dict) and d.get("term"):
            want_terms.append(str(d["term"]).strip())
        elif isinstance(d, str) and d.strip():
            want_terms.append(d.strip())

    # term -> the source line the agent says the word came from (grounding evidence).
    sent_map = {str(k).strip().lower(): str(v) for k, v in (sentences or {}).items()} \
        if isinstance(sentences, dict) else {}

    # S18 1e-core — DETERMINISTIC GROUNDING: Python OWNS the sentence. The Ask-Agent path let
    # the AI supply the source line; it frequently kept the transcript's SENTENCE FRAME but
    # swapped words ("...constructive like make it work?" for the real "...like start an
    # avalanche"). A non-verbatim sentence makes the commit-time clip lookup fall back to a
    # fuzzy surface/token match -> the audio is cut from the WRONG line. Fix: snap each term's
    # sentence to the REAL verbatim transcript line, using the AI's cited sentence only to PICK
    # among lines when several contain the term (AI proposes, Python owns). 0 AI. Degrades to
    # the agent's sentence when no transcript was ingested or the term isn't in it (gate flags).
    # Use the SRT SEGMENTS (per-subtitle-block lines, each a real cue), NOT the cached full
    # text — the latter is stored as one joined blob, so every term would "match" the whole
    # thing. The .srtpath sidecar (S18 #2) points at the timed transcript; ingest gives segments.
    try:
        from extract_vocab import _content_lemmas as _cl, _norm_match as _nm
    except Exception:
        _cl = _nm = None
    _glines, _gsegs = [], []
    try:
        from _common import load_cached_srt as _lcs
        _srtp = _lcs(source)
        if _srtp and os.path.exists(_srtp):
            from ingest_transcript import ingest_transcript as _ing
            _gsegs = [s for s in (_ing(_srtp).get("segments") or [])
                      if isinstance(s, dict) and str(s.get("text", "")).strip()]
            _glines = [str(s.get("text", "")).strip() for s in _gsegs]
    except Exception:
        _glines, _gsegs = [], []
    if not _glines:
        # S19 (B2): no srt cache (source drift / not ingested yet) — fall back to the cached
        # transcript TEXT split into SENTENCE units (same splitter the explain materials builder
        # uses) so _ground_sentence can still snap the term to a REAL line via ground_line.
        # `_gsegs` stays empty -> _seg_time_for returns {} (no fabricated timestamp: a wrong time
        # is worse than none). Additive: only ungrounded -> grounded for a genuine transcript line.
        try:
            from _common import load_cached_transcript as _lct
            _tx = _lct(source)
            if _tx:
                _glines = [ln.strip() for ln in re.split(
                    r"(?<=[.?!…])\s+|[\r\n]+|♪", _tx) if ln.strip()]
        except Exception:
            pass

    def _seg_time_for(sentence: str) -> dict:
        """S18 askfix (owner GO): timestamp for CHAT-staged words. The Mine path always carried
        start/end (via _locate_timestamp) so its infolog showed `source @ time` and its clips cut
        precisely; agent-staged words had neither (infolog missing '@', audio mislocated). The
        grounded sentence was just snapped VERBATIM onto one SRT segment, so that segment's own
        timing is exact — no fuzzy lookup. Returns {} when the sentence isn't a cue (ungrounded)
        — a wrong time is worse than none. Keys mirror the Mine unit: start/end ("HH:MM:SS" str,
        schema provenance) + start_sec/end_sec (float, millisecond-precise clip cutting)."""
        if not _gsegs or _nm is None:
            return {}
        ns = _nm(str(sentence or ""))
        if not ns:
            return {}
        for s in _gsegs:
            if _nm(str(s.get("text", ""))) == ns:
                t = {}
                if s.get("start"):
                    t["start"], t["end"] = s.get("start", ""), s.get("end", "")
                ss, se = s.get("start_sec"), s.get("end_sec")
                if isinstance(ss, (int, float)) and not isinstance(ss, bool) \
                        and isinstance(se, (int, float)) and not isinstance(se, bool):
                    t["start_sec"], t["end_sec"] = ss, se
                return t
        # S19 (owner "wash out"/"give a shot" wrong audio): a grounded sentence can SPAN several
        # srt cues ("…wash your puke | out of my jacket."), so no single segment equals it and the
        # exact match above fails. Old code returned {} -> commit RE-LOCATED via _locate_timestamp
        # and mis-hit a short cue (the "…Out!" hot-guy line). Fallback: gather, in transcript order,
        # the cues whose (>=2-word) text is contained in the sentence and span start(first)..end(last)
        # -> the real line's true window. >=2 words so a stray one-word cue ("Out!") can't pollute it.
        covering = [s for s in _gsegs
                    if (lambda st_: st_ and len(st_.split()) >= 2 and st_ in ns)(_nm(str(s.get("text", ""))))]
        if covering:
            first, last = covering[0], covering[-1]
            t = {}
            if first.get("start"):
                t["start"], t["end"] = first.get("start", ""), last.get("end", "")
            ss, se = first.get("start_sec"), last.get("end_sec")
            if isinstance(ss, (int, float)) and not isinstance(ss, bool) \
                    and isinstance(se, (int, float)) and not isinstance(se, bool):
                t["start_sec"], t["end_sec"] = ss, se
            return t
        return {}

    def _ground_sentence(term: str, ai_sentence: str) -> str:
        # S18 HEART §2/§3: snap the term to a REAL verbatim cue via the SHARED grounding helper
        # (ground_line) so Ask and Mine use ONE matcher — word-bounded verbatim, then ordered
        # lemma-run (fixes the old TIER A substring hole: "turn" matched "returns" silently, and
        # a tense-inflected phrase "turn to"↔"turned to" was missed). The AI's cited sentence only
        # DISAMBIGUATES among real cues; it never becomes the line. No cue matches -> keep the AI
        # sentence so the downstream grounding gate flags it `ungrounded` (honest flag > wrong clip).
        if not _glines:
            return ai_sentence
        from _common import ground_line
        return ground_line(term, _glines, ai_sentence) or ai_sentence

    # dedup terms (case-insensitive), preserve order
    seen, units = set(), []
    for term in want_terms:
        low = term.lower()
        if low in seen:
            continue
        seen.add(low)
        try:
            senses = wordnet_lookup(term)["senses"]
        except Exception as e:                      # OOV / lookup glitch -> still allow enrich
            log_tool_call("stage_for_review", {"term": term}, error=f"wordnet: {e}")
            senses = []
        # carry the source sentence so the enrich occurrence is grounded (S16 T1); 1e-core snaps
        # it to the real transcript line so the committed clip's audio matches the card.
        # askfix: + that cue's own start/end so chat-staged words get provenance timestamps
        # and precise clips, same as the Mine path (empty dict when ungrounded — no wrong time).
        gsent = _ground_sentence(term, sent_map.get(low, ""))
        units.append({"term": term, "sentence": gsent,
                      "senses": senses, "source": source, **_seg_time_for(gsent)})

    if units:
        # ONE batched enrich call for all fresh terms (deterministic-first; AI flags reviewable).
        staged_drafts.extend(enrich(units, source=source))

    # GROUNDING GATE (S16 T1): a draft is grounded only if some occurrence has a non-empty
    # source `sentence` AND the term actually appears in that sentence (shared content lemma).
    # This catches the "flyback" case — the agent invented/mis-recalled a word, so it has no
    # source line, OR it paired the word with a line that does not contain it. Such a node is
    # NOT staged; it is returned in `ungrounded` with a reason so the agent can re-check the
    # term against the transcript. Deterministic (no AI), additive (return only GAINS a key).
    try:
        from extract_vocab import _content_lemmas, _norm_match
    except Exception:
        _content_lemmas = None
        _norm_match = None

    # S17 ①: load the cached transcript for this source (if it was ingested this turn) so
    # we can verify a cited sentence is a REAL transcript line, not one the agent invented.
    # Deterministic (0 AI). Keyed by basename (same label reduction as `source` above).
    from _common import load_cached_transcript
    transcript = load_cached_transcript(source)
    norm_transcript = _norm_match(transcript) if (transcript and _norm_match) else ""
    # S19 (B1): the cached transcript is ONE space-joined blob (ingest joins segments with " "),
    # so measuring token overlap against the whole blob let a fabricated sentence made of common
    # words score ~1.0 (every token appears somewhere in the film) and pass ungrounded — invented
    # lines reached Anki. Split the blob into SENTENCE units (same splitter as _materials_for) and
    # check overlap PER UNIT: a real (re-wrapped / OCR-noisy) line still matches one unit highly,
    # but an invented line whose words are scattered across the film never reaches the threshold.
    _tx_units = [u for u in (_norm_match(s) for s in re.split(
        r"(?<=[.?!…])\s+|[\r\n]+|♪", transcript)) if u] \
        if (transcript and _norm_match) else []

    def _transcript_hint(term: str) -> str:
        """Real transcript line(s) containing the term — handed to the learner/agent so an
        ungrounded row can be corrected in ONE step (paste the suggested line) instead of digging
        through the script. S19 (#4a): the cached transcript is ONE space-joined blob, so the old
        split on \\r\\n returned the WHOLE transcript as a single 'line' (useless). Split into
        SENTENCE units (same splitter as B1) and rank by how many of the term's content lemmas the
        sentence contains, so the BEST real candidate line surfaces first."""
        if not transcript or _content_lemmas is None:
            return ""
        tl = _content_lemmas(term)
        if not tl:
            return ""
        sents = [s.strip() for s in re.split(r"(?<=[.?!…])\s+|[\r\n]+|♪", transcript) if s.strip()]
        scored = [(len(tl & _content_lemmas(s)), s) for s in sents]
        scored = sorted(((n, s) for n, s in scored if n), key=lambda x: -x[0])
        hits = [s for _, s in scored[:2]]
        return (" Real transcript line(s): " + " | ".join(hits)) if hits else ""

    def _sentence_in_transcript(sentence: str) -> bool:
        """True if the cited sentence is (substantially) a REAL line in the transcript.
        S19 (B1): overlap is measured per SENTENCE UNIT (max), NOT against the whole blob."""
        if not norm_transcript or _norm_match is None:
            return True                        # no transcript cached -> cannot check here
        ns = _norm_match(sentence)
        if not ns:
            return False
        if ns in norm_transcript:              # exact normalized substring -> real line
            return True
        # tolerate whisper/OCR noise & re-wrapping: high token overlap with a SINGLE real unit
        # counts as grounded; a fabricated sentence spread across many units never qualifies.
        toks = set(ns.split())
        if not toks:
            return False
        for u in _tx_units:
            ut = set(u.split())
            if ut and len(toks & ut) / len(toks) >= 0.8:
                return True
        return False

    def _grounded_reason(node: dict) -> str:
        """"" if grounded, else a human/agent-readable reason it is not.
        Two tiers: (1) the word must appear in the cited sentence; (2) the cited sentence
        must be a real transcript line (checked only when a transcript was ingested)."""
        occs = [o for o in (node.get("occurrences") or [])
                if isinstance(o, dict) and str(o.get("sentence", "")).strip()]
        term = node.get("term", "")
        if not occs:
            # No source line at all. If the agent staged WITHOUT ingesting a transcript,
            # that IS the fabrication signal — tell it to ingest first (don't spin a loop).
            if not transcript:
                return ("no source sentence, and no transcript has been ingested — "
                        "call ingest_transcript on the media FIRST, then keep words with "
                        "their exact source line")
            return ("no grounded occurrence (no source sentence) — "
                    "re-check the term against the transcript" + _transcript_hint(term))
        if _content_lemmas is None or not term:
            return ""                          # cannot verify overlap -> don't block (degrade)
        term_lemmas = _content_lemmas(term)
        if not term_lemmas:
            return ""                          # all-stopword term -> nothing to verify against
        # TIER 1 — the word must be in the cited sentence.
        word_in_sent = [o for o in occs
                        if term_lemmas & _content_lemmas(o.get("sentence", ""))]
        if not word_in_sent:
            return ("the word does not appear in the source sentence you gave — "
                    "quote the exact transcript line that contains this term"
                    + _transcript_hint(term))
        # TIER 2 (S17 ①) — the cited sentence must be a REAL transcript line, not invented.
        if norm_transcript and not any(
                _sentence_in_transcript(o.get("sentence", "")) for o in word_in_sent):
            return ("the sentence you cited is not in the transcript (it looks invented) — "
                    "quote the exact line from the transcript" + _transcript_hint(term))
        return ""

    # S17 (owner decision, supersedes the S16 hard-drop): an ungrounded term is STILL STAGED,
    # but FLAGGED — "AI proposes, Python detects, the HUMAN decides". Only the learner may
    # reject a word; Python's job is to make the problem visible (⚠ ai flag = 'ungrounded'),
    # and the commit backstop (validate_edits) still blocks it until the learner supplies a
    # real source sentence in the review table. `ungrounded` is still returned so the agent
    # tells the learner exactly which words need a checked sentence.
    ungrounded, seen_ung = [], set()
    for d in staged_drafts:
        node = d.get("node", {}) if isinstance(d, dict) else {}
        reason = _grounded_reason(node)
        if reason:
            d["needs_review"] = True
            # S19 OPEN-3: PERSIST the reason (it carries "Real transcript line(s): …" from
            # _transcript_hint). Before, only the chat `ungrounded` list got the reason, so the
            # Review tab's banner + ⚠ column had the flag but no suggested line. Stashing it lets
            # pending_to_rows surface the hint where the learner fixes the row.
            d["ungrounded_reason"] = reason
            flags = [f for f in (d.get("ai_fields") or [])]
            if "ungrounded" not in flags:
                flags.append("ungrounded")
            d["ai_fields"] = flags
            term = node.get("term", "") if isinstance(node, dict) else ""
            low = term.lower()
            if term and low not in seen_ung:
                seen_ung.add(low)
                ungrounded.append({"term": term, "reason": reason})

    if not staged_drafts:
        out = {"staged": [], "newly_staged": [], "already_present": [], "updated": [],
               "ungrounded": ungrounded, "review_path": review_io.PENDING_PATH}
        log_tool_call("stage_for_review", args, result=out)
        return out

    # Which of these were ALREADY in the queue? export_review(mode="append") dedups by key and
    # KEEPS an existing entry, so re-staging an already-queued word adds NO new "undecided" row.
    # Snapshot the existing keys BEFORE the append so the agent can tell the learner exactly what
    # is new vs. already saved — otherwise "staged N" over-reports the duplicates. The pending
    # stash (pending_drafts.json) is now the single review queue, so its keys are the truth.
    existing_pending = review_io.load_pending()
    existing_keys = {k for k in existing_pending if k != "_meta"}
    # S18 #1/#4 — a key already in the queue is "updated" (not merely "already present") when
    # the OLD row was flagged `ungrounded` and THIS draft is grounded: export_review now
    # replaces it with the corrected sentence. Snapshot which existing keys are ungrounded so
    # the agent can honestly say "updated the sentence" vs "already saved, unchanged" (#4).
    ungrounded_keys = {k for k, v in existing_pending.items()
                       if k != "_meta" and isinstance(v, dict)
                       and "ungrounded" in (v.get("ai_fields") or [])}

    review_io.export_review(staged_drafts, mode="append", source=source)

    newly, already, updated, seen = [], [], [], set()
    for d in staged_drafts:
        if not isinstance(d, dict):
            continue
        node = d.get("node", {})
        term, key = node.get("term", ""), node.get("key", "")
        if not term or key in seen:
            continue
        seen.add(key)
        if key not in existing_keys:
            newly.append(term)
        elif key in ungrounded_keys and "ungrounded" not in (d.get("ai_fields") or []):
            updated.append(term)                # re-staged with a corrected, grounded sentence
        else:
            already.append(term)

    staged = [d.get("node", {}).get("term", "") for d in staged_drafts if isinstance(d, dict)]
    out = {"staged": [s for s in staged if s], "newly_staged": newly,
           "already_present": already, "updated": updated, "ungrounded": ungrounded,
           "review_path": review_io.PENDING_PATH}
    log_tool_call("stage_for_review", args, result=out)
    return out


if __name__ == "__main__":
    import json
    import sys
    ts = sys.argv[1:] or ["fed up"]
    print(json.dumps(stage_for_review(terms=ts, source="cli"), ensure_ascii=False, indent=2))
