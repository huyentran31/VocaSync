"""
S17 ① — sentence∈transcript grounding on the agent write path (stage_for_review).

The pre-existing gate only checked word∈sentence, so the agent could INVENT a sentence
that happened to contain the word and it sailed through. This adds a second tier: when a
transcript was ingested, the cited sentence must be a REAL transcript line.

Proves, offline (mocked AI, 0 API calls):
  • a term whose sentence IS a real transcript line  -> grounded (not flagged),
  • a term whose sentence contains the word but is NOT in the transcript (fabricated)
    -> STAGED but FLAGGED 'ungrounded', with the real transcript line in the reason,
  • NO transcript cached -> tier-2 is skipped (degrade): a word-in-sentence term is NOT
    flagged just because we cannot verify it (no false positives on the pure Q&A path).
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "agent"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import enrich as en
import review_io
from _common import GRAPH_PATH, cache_transcript, _transcript_cache_path, _srtpath_cache_path

SOURCE = "TestFilm"
TRANSCRIPT = (
    "How did you find out about the meeting?\n"
    "I'm fed up with all these delays.\n"
    "Let's go on with the plan.\n"
)


def _stash(path):
    if os.path.exists(path):
        bak = path + ".testbak"
        shutil.move(path, bak)
        return bak
    return None


def _restore(path, bak):
    if os.path.exists(path):
        os.remove(path)
    if bak and os.path.exists(bak):
        shutil.move(bak, path)


def _run(terms, sentences, cache=True, transcript_text=None):
    import stage_for_review as sfr
    config.AI_API_KEY = "fake"
    en.call_ai = lambda p, s: json.dumps(
        [{"term": t, "sense_id": "", "confidence": 0.8} for t in terms])
    if cache:
        cache_transcript(SOURCE, transcript_text if transcript_text is not None else TRANSCRIPT)
    return sfr.stage_for_review(terms=terms, source=SOURCE, sentences=sentences)


def test_real_line_grounded_fabricated_flagged():
    """S17 gate + S19 (B2): with a transcript cached (no srt) —
      • a term cited with its REAL line stays grounded;
      • a term whose WORD is genuinely in the transcript but was cited with a hallucinated frame
        is now SNAPPED to its real line via the B2 transcript-text fallback (Python owns the
        sentence) — grounded, not flagged (this used to only happen when an srt was cached);
      • a term whose word is ABSENT from the transcript can't be snapped -> STAGED but FLAGGED
        ungrounded, so genuinely invented usage still can't sail through to Anki."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    sbak = _stash(_srtpath_cache_path(SOURCE))    # exercise the NO-srt (B2) path deterministically
    try:
        out = _run(
            terms=["fed up", "go on", "kick the bucket"],
            sentences={
                "fed up": "I'm fed up with all these delays.",        # REAL line -> grounded
                "go on": "You should go on a long vacation soon.",    # word IS in transcript -> B2 snaps
                "kick the bucket": "He almost kicked the bucket.",    # word ABSENT -> flagged ungrounded
            },
        )
        assert set(out["staged"]) >= {"fed up", "go on", "kick the bucket"}, out
        ung = {u["term"]: u["reason"] for u in out.get("ungrounded", [])}
        assert "fed up" not in ung, ("real transcript line must NOT be flagged", out)
        assert "go on" not in ung, ("B2 must snap an in-transcript word to its real line", out)
        assert "kick the bucket" in ung, ("word absent from transcript must be flagged", out)
        # B2 replaced the hallucinated frame with the real verbatim cue
        pend = review_io.load_pending()
        by_term = {v["node"]["term"]: v["node"]["occurrences"][0]["sentence"]
                   for k, v in pend.items() if k != "_meta" and isinstance(v, dict)}
        assert by_term["go on"] == "Let's go on with the plan.", ("snapped to real line", by_term)
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE),
                     _srtpath_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
        _restore(_srtpath_cache_path(SOURCE), sbak)
    print("grounding OK -> real line grounded, in-transcript word B2-snapped, absent word flagged")


def test_no_transcript_degrades():
    """No transcript cached -> we cannot check sentence∈transcript, so a word-in-sentence
    term is NOT flagged (degrade, no false positive). Same behaviour the pure Q&A path relies on."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    try:
        out = _run(
            terms=["go on"],
            sentences={"go on": "You should go on a long vacation soon."},
            cache=False,
        )
        ung = [u["term"] for u in out.get("ungrounded", [])]
        assert "go on" not in ung, ("no transcript -> tier-2 must be skipped (degrade)", out)
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
    print("grounding OK -> no transcript degrades cleanly (no false positive)")


def test_restage_updates_ungrounded_sentence():
    """S18 #1 — re-staging an ungrounded word with the REAL transcript line must UPDATE the
    queue row (old bug: dedup-by-key kept the fabricated sentence, so 'updated it' was a lie).
    S19 (B2): because an in-transcript word now auto-snaps on the FIRST stage, the update path is
    exercised the way it is actually reached — a word the transcript does NOT yet contain stays
    ungrounded, then the learner re-grounds it after the RIGHT transcript is cached. Proves:
    (1) first stage (word absent from the cached transcript) is flagged ungrounded, storing the
    fabricated sentence; (2) with the correct transcript cached, re-staging with the real line
    clears the flag, replaces the sentence, and reports the term under `updated`."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    sbak = _stash(_srtpath_cache_path(SOURCE))
    NO_GOON = "How did you find out about the meeting?\nI'm fed up with all these delays.\n"
    FAKE = "You should go on a long vacation soon."       # 'go on' NOT in NO_GOON -> ungrounded
    REAL = "Let's go on with the plan."                   # real line in the corrected transcript
    try:
        out1 = _run(terms=["go on"], sentences={"go on": FAKE},
                    transcript_text=NO_GOON)
        assert "go on" in [u["term"] for u in out1.get("ungrounded", [])], out1
        pend = review_io.load_pending()
        key = next(k for k in pend if k != "_meta")
        assert pend[key]["node"]["occurrences"][0]["sentence"] == FAKE, pend[key]

        out2 = _run(terms=["go on"], sentences={"go on": REAL})    # default transcript HAS the line
        assert "go on" in out2.get("updated", []), ("re-stage must report updated", out2)
        assert "go on" not in out2.get("already_present", []), out2
        pend2 = review_io.load_pending()
        assert pend2[key]["node"]["occurrences"][0]["sentence"] == REAL, ("queue sentence must be corrected", pend2[key])
        assert "ungrounded" not in (pend2[key].get("ai_fields") or []), ("flag must clear", pend2[key])
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE),
                     _srtpath_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
        _restore(_srtpath_cache_path(SOURCE), sbak)
    print("restage OK -> corrected sentence replaces fabricated one, flag cleared, reported as updated")


def test_1e_snaps_sentence_to_real_srt_line():
    """S18 1e-core — when an SRT is cached, Python OWNS the sentence: a hallucinated line that
    keeps the transcript's frame but swaps words is REPLACED with the real verbatim cue, so the
    committed clip's audio matches the card. A phrase that is NOT in the transcript is left for
    the gate to flag (not mis-snapped)."""
    import tempfile
    from _common import cache_transcript, load_cached_srt
    srt = ("1\n00:00:10,000 --> 00:00:12,000\nI've tried to make it work, really.\n\n"
           "2\n00:00:20,000 --> 00:00:22,000\nThat's no reason to get a divorce.\n")
    d = tempfile.mkdtemp()
    srt_path = os.path.join(d, "film.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt)
    full = "I've tried to make it work, really. That's no reason to get a divorce."
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    try:
        import stage_for_review as sfr
        config.AI_API_KEY = "fake"
        en.call_ai = lambda p, s: json.dumps([{"term": "make it work", "sense_id": "", "confidence": 0.8},
                                              {"term": "vacation", "sense_id": "", "confidence": 0.8}])
        cache_transcript(SOURCE, full, srt_path=srt_path)
        assert load_cached_srt(SOURCE) == srt_path, load_cached_srt(SOURCE)
        out = sfr.stage_for_review(
            terms=["make it work", "vacation"], source=SOURCE,
            sentences={"make it work": "Can't you do something like make it work?",  # hallucinated frame
                       "vacation": "I really need a long vacation."})                # word NOT in srt
        pend = review_io.load_pending()
        by_term = {v["node"]["term"]: v for k, v in pend.items()
                   if k != "_meta" and isinstance(v, dict)}
        # 1e snapped the hallucinated line to the REAL verbatim cue
        assert by_term["make it work"]["node"]["occurrences"][0]["sentence"] == \
            "I've tried to make it work, really.", by_term["make it work"]["node"]["occurrences"][0]
        # "vacation" isn't in the transcript at all -> not mis-snapped -> flagged ungrounded
        assert "vacation" in [u["term"] for u in out.get("ungrounded", [])], out
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
    print("1e OK -> hallucinated sentence snapped to real SRT cue; absent phrase flagged")


def test_materials_for_verbatim_wins_and_absent_term_dropped():
    """S18 — _materials_for (explain 'From this video' grounding) must not repeat the P0-1b
    bug: a multi-word phrase ("turn to") used to match the FIRST line sharing ANY content
    lemma, so it quoted a garbled line containing just "turn". Now: (1) a verbatim phrase
    match beats an earlier lemma-overlap line; (2) a term with no verbatim/full-lemma line
    is DROPPED (explain says 'not found in this video' instead of quoting a wrong line)."""
    from loop import _materials_for
    cbak = _stash(_transcript_cache_path(SOURCE))
    garbled = "It is infuriating that your unhappiness does not turn too fat."
    real = "You can always turn to me for help."
    try:
        cache_transcript(SOURCE, garbled + "\n" + real + "\nI'm fed up with all these delays.\n")
        mat = _materials_for("turn to, fed up, banana split", SOURCE)
        by_term = {h["term"]: h["line"] for h in mat.get("hits", [])}
        assert by_term.get("turn to") == real, ("verbatim phrase line must win", mat)
        assert by_term.get("fed up") == "I'm fed up with all these delays.", mat
        assert "banana split" not in by_term, ("absent term must be dropped, not guessed", mat)
    finally:
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    print("materials_for OK -> verbatim line wins over lemma-overlap; absent term dropped")


def test_ground_line_verbatim_inflection_and_no_false_substring():
    """S18 HEART §3 — the SHARED grounding helper. (a) word-bounded verbatim phrase wins;
    (b) a tense-inflected variant is caught via the ordered lemma-run ("turn to" ↔ "turned
    to"); (c) a bare substring is NOT a match ("returns" must not ground "turn"); (d) a
    scramble ("turn too fat") does not match "turn to"; (e) a garbled/absent term -> None."""
    from _common import ground_line
    real = "You can always turn to me for help."
    garbled = "It does not turn too fat, sadly today."
    # (a) verbatim phrase beats the earlier garbled line that only shares the word "turn"
    assert ground_line("turn to", [garbled, real]) == real, "verbatim phrase must win"
    # (b) inflected variant via lemma-run
    assert ground_line("turn to", ["I turned to leave the room."]) == "I turned to leave the room."
    # (c) substring without boundary must NOT match ("returns" != "turn")
    assert ground_line("turn", ["She returns home now."]) is None, "returns must not ground turn"
    # (d) scrambled run must not match the ordered phrase
    assert ground_line("turn to", [garbled]) is None, "scramble must not match"
    # (e) absent term -> None (caller says 'Not found in this video.')
    assert ground_line("banana split", [real, garbled]) is None
    # (f) askfix TIER 3 — separable phrasal verb: "put down" grounds "...put her down." (pronoun
    # object inserted) but NOT the compound "put the down payment" (determiner gap rejected).
    assert ground_line("put down", ["It tells you when to put her down."]) == \
        "It tells you when to put her down.", "separable phrasal (pronoun gap) must match"
    assert ground_line("turn off", ["Can you turn it off, please?"]) == "Can you turn it off, please?"
    assert ground_line("put down", ["He put the down payment on the house."]) is None, \
        "determiner-gap compound must NOT falsely match a phrasal verb"
    # (g) askfix V4 — reflexive pronouns normalize: dictionary form "knock oneself out"
    # must ground the real spoken cue "knock yourself out."
    assert ground_line("knock oneself out", ["knock yourself out."]) == "knock yourself out."
    # (h) askfix S19 TIER 4 — -ing inflection, contiguous: "go solo" ↔ "going solo"
    assert ground_line("go solo", ["Come on, no one's going solo on this."]) == \
        "Come on, no one's going solo on this.", "-ing inflection must ground (go solo <-> going solo)"
    # (i) askfix S19 TIER 5 — inserted filler / inflection in MULTI-word (>=3-token) terms
    assert ground_line("take the wrong way", ["Mom, don't take this the wrong way,"]) == \
        "Mom, don't take this the wrong way,", "inserted 'this' must still ground"
    assert ground_line("eat out of house and home", ["He's eating me out of house and home"]) == \
        "He's eating me out of house and home", "inserted 'me' + -ing must ground"
    # (j) the gap tier is FILLERS-only: a CONTENT-word gap must NOT match
    assert ground_line("take the wrong way", ["Please take this the completely wrong way."]) is None, \
        "a content-word gap ('completely') between term tokens must not match"
    # (k) the >=3-token gap tier must NOT re-open determiner gaps for 2-word phrasals
    assert ground_line("put down", ["He put the down payment on the house."]) is None, \
        "2-word phrasal still rejects a determiner gap (unchanged by TIER 5)"
    print("ground_line OK -> verbatim/inflection/separable + S19 -ing & filler-gap; substring/scramble/content-gap rejected")


def test_ask_fabrication_guard_replaces_bad_from_this_video_quotes():
    """S18 HEART §1b — the Ask fabrication backstop. Given a fake transcript + a fake explain
    output whose 'From this video' section INVENTS film lines, after Python's post-check EVERY
    quoted line in that section is either verbatim-in-transcript or exactly 'Not found in this
    video.' — and text OUTSIDE the section is left untouched. 0 AI, deterministic."""
    from loop import _verify_video_quotes
    cbak = _stash(_transcript_cache_path(SOURCE))
    transcript = "Okay, fire away.\nNo can do, sis.\nHere we go.\n"
    try:
        cache_transcript(SOURCE, transcript)
        mats = {"source": SOURCE, "hits": [{"term": "fire away", "line": "Okay, fire away."}]}
        text = (
            "**From this video: %s**\n" % SOURCE
            + '- "fire away": "Fire away with your questions!"\n'   # fabricated, term IS grounded
            + '- "put down": "Please put down the vase gently."\n'  # fabricated, term NOT in video
            + "\n**Dictionary**\n"
            + '- unrelated example: "Fire away with your questions!"\n'  # OUTSIDE section -> untouched
        )
        cues = {h["term"]: h["line"] for h in mats["hits"]}
        out = _verify_video_quotes(text, [SOURCE], cues, [])
        video_part = out.split("**Dictionary**")[0]
        # grounded term -> replaced with its REAL cue; the fabricated line is gone from the section
        assert "Okay, fire away." in video_part, out
        assert "Fire away with your questions!" not in video_part, ("fabricated line must be gone", out)
        # ungrounded absent term -> honest 'Not found in this video.'
        assert "Not found in this video." in video_part, out
        assert "Please put down the vase gently." not in out, ("fabricated absent line must be gone", out)
        # OUTSIDE the section: the Dictionary example quote is left exactly as-is
        assert 'unrelated example: "Fire away with your questions!"' in out, ("outside section untouched", out)

        # live Q2 regression: (a) an INLINE label line's quote IS policed; (b) a Meaning/Example
        # line inside the section is an EXPLANATION, not a provenance claim -> untouched.
        text2 = ('### 1. No can do\n'
                 '**Meaning:** an informal way of saying "I cannot do that, sorry."\n'
                 '**From this video: %s:** "No can do, buddy, not this time."\n' % SOURCE
                 + '*   *Example 1:* "I asked nicely but she said no can do again."\n')
        out2 = _verify_video_quotes(text2, [SOURCE], {}, [])
        assert '"I cannot do that, sorry."' in out2, ("Meaning line must be untouched", out2)
        assert '"I asked nicely but she said no can do again."' in out2, ("Example line untouched", out2)
        assert '"No can do, buddy, not this time."' not in out2, ("inline label quote policed", out2)
        assert "Not found in this video." in out2, out2
    finally:
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    print("ask fabrication-guard OK -> bad 'From this video' quotes replaced; outside untouched")


def test_guard_uses_recall_lines_on_followup_turn():
    """S18 askfix (fix 2) — the fabrication guard must work on a RECALL-ONLY follow-up turn
    (nothing ingested this turn), using the real occurrence lines recall surfaced as ground
    truth. Live bug: asked about 'get a say' in a later turn, the agent invented the film line
    'I should get a say in this, too' under 'From your graph' — the real line is 'So what, the
    Vibe doesn't get a say?'. With NO cached transcript, the guard still (a) keeps a real line,
    (b) replaces the fabricated one with the real cue for a term named on the line."""
    from loop import _verify_video_quotes
    real = "So what, the Vibe doesn't get a say?"
    text = (
        "### get a say\n"
        "**From your graph**\n"
        '- You previously saw this in Charade: "I should get a say in this, too."\n'  # FABRICATED
        "**Dictionary**\n"
        '- **Examples:** "Everyone should get a say in the decision."\n'             # explanation, keep
    )
    out = _verify_video_quotes(text, [], {"get a say": real}, [real])
    assert "I should get a say in this, too." not in out, ("fabricated graph line must be gone", out)
    assert real in out, ("real cue must replace it", out)
    assert '"Everyone should get a say in the decision."' in out, ("example untouched", out)
    print("followup guard OK -> fabricated 'From your graph' line replaced from recall truth")


def test_chat_stage_carries_timestamp():
    """S18 askfix (owner GO) — a CHAT-staged word must carry the timestamp of its grounded SRT
    cue (start/end + start_sec/end_sec), same as the Mine path, so the infolog shows `@ time`
    and the clip cuts precisely. An ungrounded sentence gets NO time (wrong time is worse)."""
    import tempfile
    from _common import cache_transcript, load_cached_srt
    srt = ("1\n00:00:10,000 --> 00:00:12,000\nIt tells you when to put her down.\n\n"
           "2\n00:01:00,000 --> 00:01:02,000\nOkay, fire away.\n")
    d = tempfile.mkdtemp()
    srt_path = os.path.join(d, "film.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt)
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    try:
        import stage_for_review as sfr
        config.AI_API_KEY = "fake"
        en.call_ai = lambda p, s: json.dumps(
            [{"term": "fire away", "sense_id": "", "confidence": 0.8},
             {"term": "vacation", "sense_id": "", "confidence": 0.8}])
        cache_transcript(SOURCE, "It tells you when to put her down. Okay, fire away.",
                         srt_path=srt_path)
        assert load_cached_srt(SOURCE) == srt_path
        sfr.stage_for_review(terms=["fire away", "vacation"], source=SOURCE,
                             sentences={"fire away": "Okay, fire away.",
                                        "vacation": "I need a long vacation."})   # not in srt
        pend = review_io.load_pending()
        by_term = {v["node"]["term"]: v["node"]["occurrences"][0]
                   for k, v in pend.items() if k != "_meta" and isinstance(v, dict)}
        occ = by_term["fire away"]
        assert occ.get("start") == "00:01:00", ("grounded cue's own start", occ)
        assert occ.get("end") == "00:01:02", occ
        assert not by_term["vacation"].get("start"), ("ungrounded word must carry NO time", by_term["vacation"])
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
    print("chat-stage timestamp OK -> grounded cue time carried; ungrounded stays untimed")


def test_infolog_chronological_with_definition():
    """S18 askfix (owner) — infolog orders entries by (source, first timestamp) so it reads
    like the film, and each word shows its saved definition ('= ...') under the term line."""
    import tempfile
    from infolog_export import export_infolog
    nodes = [
        {"term": "zebra", "word_type": "word", "definition": "striped animal",
         "occurrences": [{"source": "F.srt", "start": "00:00:05", "sentence": "A zebra runs."}]},
        {"term": "apple", "word_type": "word", "definition": "a fruit",
         "occurrences": [{"source": "F.srt", "start": "00:10:00", "sentence": "An apple falls."}]},
        {"term": "mango", "word_type": "word", "definition": "",
         "occurrences": [{"source": "F.srt", "start": "", "sentence": "Mango time."}]},
    ]
    out = os.path.join(tempfile.mkdtemp(), "infolog.txt")
    export_infolog(nodes, out)
    text = open(out, encoding="utf-8").read()
    # chronological: zebra (00:00:05) BEFORE apple (00:10:00); untimed mango last
    assert text.index("zebra") < text.index("apple") < text.index("mango"), text
    assert "= striped animal" in text and "= a fruit" in text, ("definitions shown", text)
    assert "F.srt @ 00:00:05" in text, text
    print("infolog OK -> chronological by timestamp; definitions included; untimed last")


def test_summary_block():
    """S18 askfix (owner #1) — the deterministic end-of-turn summary. Discovery turn: report
    total found, the already-saved ones, and the un-explained remainder (offer to continue).
    Plain known-word turn (no discovery): just the review pointer. Nothing: empty."""
    from loop import _summary_block
    # S19 (#1): buckets are MUTUALLY EXCLUSIVE — 'fire away' is BOTH explained-this-turn AND already
    # in the graph, but must be counted ONLY as explained (old bug double-listed it, so the counts
    # didn't reconcile). explained(2) + known-not-explained(2) + remaining(1) == found(5).
    cands = [{"term": "fire away"}, {"term": "no can do"}, {"term": "get a say"},
             {"term": "put down"}, {"term": "hang out"}]
    known = {"fire away": "your collection", "get a say": "your collection", "put down": "review queue"}
    out = _summary_block(cands, explained_terms=["fire away", "no can do"], known_terms=known)
    assert "Found **5**" in out and "Explained **2**" in out, out
    # owner: say WHERE — graph vs review queue are different things
    assert "in your graph): get a say" in out, out
    # 'fire away' was explained this turn -> must NOT also appear in the "already learned" line
    assert "in your graph): get a say" in out and "fire away" not in out.split("Not explained")[0].split("in your graph)")[1].split("\n")[0], \
        ("explained term must not be double-listed as already-learned", out)
    assert "REVIEW QUEUE" in out and "put down" in out, ("queue words named separately", out)
    assert "Not explained yet (1): hang out" in out, ("remaining lists un-explained new terms", out)
    assert "explain the rest" in out, out
    # plain known-word question (no discovery) -> review pointer, split by location
    out2 = _summary_block([], explained_terms=["knock yourself out"],
                          known_terms={"knock yourself out": "review queue"})
    assert out2.startswith("\n\n---\n\n💡 **Review it:**") and "knock yourself out" in out2, out2
    assert "review queue" in out2 and "Found" not in out2, out2
    # nothing to say -> empty
    assert _summary_block([], [], {}) == "", "no data -> no block"
    print("summary block OK -> found/known/remaining on discovery; review pointer on known-word; empty otherwise")


def test_format_explain_fields_ungluing():
    """S19 (#2) — field labels the AI ran onto one line (or joined with a single newline, which
    Markdown renders as a space) get their own line via a BLANK line; content untouched and prose
    that merely contains a label word ('the register of') is NOT split."""
    from loop import _format_explain_fields
    # (a) same-line glue -> each label starts its own line (blank line before it)
    glued = ("**Meaning:** to earn money. Examples: She works hard. Synonyms: earn a living. "
             "Register: Neutral. Pronunciation: /liv/")
    out = _format_explain_fields(glued)
    for lbl in ("Examples:", "Synonyms:", "Register:", "Pronunciation:"):
        assert ("\n\n" + lbl) in out, (f"{lbl} must start its own line", out)
    assert "earn a living" in out and "/liv/" in out, ("content preserved", out)
    # (b) a SINGLE newline (renders as a space in Markdown) is promoted to a blank line
    assert _format_explain_fields("x\nRegister: y") == "x\n\nRegister: y", "single newline -> blank line"
    # (c) already a blank line -> unchanged (idempotent)
    assert _format_explain_fields("x\n\nRegister: y") == "x\n\nRegister: y", "idempotent on blank line"
    # (d) lowercase prose word that is not a labelled field is untouched
    prose = "This changes the register of the sentence."
    assert _format_explain_fields(prose) == prose, ("lowercase prose word must not split", prose)
    print("format explain OK -> labels get own line (space/newline -> blank line); prose + idempotent safe")


def test_a1_save_all_stages_full_found_set():
    """S19 (#1) — 'save all' stages the FULL discovered set (from the ALL_FOUND marker),
    deterministically (0 LLM), not an LLM-guessed subset. This is what the 'Save all found'
    button triggers; it fixes the live 'saved 14 of 28' miscount."""
    cbak = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "No can do, sis. You have to blend in. Okay, fire away.")
    staged = {"newly_staged": ["no can do", "blend in", "fire away"], "already_present": [], "ungrounded": []}

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "stage_for_review":
            assert set(args["terms"]) == {"no can do", "blend in", "fire away"}, ("full set staged", args)
            return staged
        raise AssertionError(f"unexpected tool {name} (save-all must not explain)")

    loop, counters, restore = _patch_agent([], tools)   # NO ai responses -> any call_ai fails
    prior = ["USER QUERY: find phrases",
             f"ATTACHED SOURCE FILE: {SOURCE}",
             "REMAINING_UNEXPLAINED: ",                       # everything already explained
             "ALL_FOUND: no can do, blend in, fire away"]
    try:
        out = loop.run_agent("save all the found phrases to my queue", prior_scratch=prior)
    finally:
        restore()
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    assert counters["ai"] == 0, ("save-all must be Python-driven — 0 LLM", counters)
    traj = [t for t, _ in counters["tools"]]
    assert "stage_for_review" in traj and "explain" not in traj, ("must stage the full set, not explain", traj)
    assert "review queue" in out["answer"].lower(), out["answer"]
    print("A1 save-all OK -> full ALL_FOUND set staged deterministically (0 LLM), not a guessed subset")


def test_transcript_hint_returns_real_sentence_not_blob():
    """S19 (#4a) — the ungrounded hint must return the actual REAL sentence (blob split into
    sentence units + ranked by lemma overlap), not the whole space-joined transcript."""
    import stage_for_review as sfr
    import config as _cfg
    cbak = _stash(_transcript_cache_path(SOURCE))
    sbak = _stash(_srtpath_cache_path(SOURCE))
    TX = ("How did you find out? He kills bugs for a living. He's eating me out of house and home. "
          "Just try and blend in.")
    try:
        _cfg.AI_API_KEY = "fake"
        en.call_ai = lambda p, s: json.dumps([{"term": "eat out of house and home", "sense_id": "", "confidence": 0.8}])
        cache_transcript(SOURCE, TX)
        out = sfr.stage_for_review(
            terms=["eat out of house and home"], source=SOURCE,
            sentences={"eat out of house and home": "We ate them out of house and home last week."})
        ung = {u["term"]: u.get("reason", "") for u in out.get("ungrounded", [])}
        # B2 may snap it grounded; if flagged, the reason must name the REAL sentence, not the blob
        if "eat out of house and home" in ung:
            r = ung["eat out of house and home"]
            assert "He's eating me out of house and home." in r, ("hint must surface the real line", r)
            assert TX.strip() not in r, ("hint must NOT be the whole transcript blob", r)
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE),
                     _srtpath_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(_transcript_cache_path(SOURCE), cbak)
        _restore(_srtpath_cache_path(SOURCE), sbak)
    print("transcript hint OK -> suggests the real sentence, not the whole blob")


def test_renumber_terms_across_batches():
    """S18 askfix (owner #5) — chained explain batches each number 1..5; the joined answer must
    count 1..N continuously. Both heading shapes renumbered; bullets/prose untouched."""
    from loop import _renumber_terms
    joined = ("### 1. at the last minute\ntext\n### 2. pest control\ntext\n\n---\n\n"
              "1. **take the wrong way**\n*   *Example:* \"one 1. two\"\n2. **put down**\ntext")
    out = _renumber_terms(joined)
    assert "### 1. at the last minute" in out and "### 2. pest control" in out, out
    assert "3. **take the wrong way**" in out, ("second batch continues at 3", out)
    assert "4. **put down**" in out, out
    assert '*   *Example:* "one 1. two"' in out, ("bullets/prose untouched", out)
    print("renumber OK -> continuous 1..N across batches; content untouched")


def test_guard_line_local_does_not_touch_notes():
    """askfix — per-term layout regression: the guard must police ONLY lines that themselves
    claim a source ('From this video'/'From your graph') + cue bullets under a source HEADING,
    never a Meaning/Note/Definition line. (Live bug: 'an idiom for Not found in this video.' and
    'definition: Not found in this video.' — the guard bled across a whole per-term block.)"""
    from loop import _verify_video_quotes
    text = (
        "### 1. knock yourself out\n"
        "**Meaning:** feel free to go ahead.\n"
        "**From this video: F.srt:** \"There's a sandwich in the fridge knock yourself out.\"\n"
        "**Examples:**\n"
        "- \"If you want to try it, knock yourself out.\"\n"
        "**Note:** Do not take it literally; it is strictly an idiom for permission.\n"
        "### From this video: F.srt\n"                      # block-format heading + cue bullet
        "- fire away: \"Fire away with your questions!\"\n"  # fabricated cue -> must be replaced
    )
    real = "There's a sandwich in the fridge knock yourself out."
    out = _verify_video_quotes(text, sources=[], cues={"fire away": "Okay, fire away."},
                               real_lines=[real])
    # the real inline 'From this video' quote is kept
    assert real in out, ("real inline source line kept", out)
    # the Note line is UNTOUCHED (no bleed) — the exact live-bug string must NOT appear
    assert "strictly an idiom for permission." in out, ("Note line must be untouched", out)
    assert "idiom for Not found" not in out, ("guard must not corrupt the Note", out)
    # the Example bullet (under an Examples: label, not a source heading) is untouched
    assert '"If you want to try it, knock yourself out."' in out, ("example bullet untouched", out)
    # the block-format fabricated cue bullet IS replaced with the real cue
    assert "Fire away with your questions!" not in out, ("block-format fabricated cue replaced", out)
    assert "Okay, fire away." in out, ("replaced with real cue", out)
    print("guard line-local OK -> only source lines/cue bullets policed; notes/examples untouched")


def test_finish_dedup_drops_recap_keeps_followup():
    """S18 HEART §1e — the model's trailing `final` must not re-print what explain already
    said. (a) a verbatim re-dump of the batches -> dropped entirely; (b) a PARAPHRASED
    per-term recap ('**term**: gloss' lines under '### Batch N', live Q1 shape) -> those lines
    and their emptied headings pruned, while a genuinely NEW note + follow-up question survive."""
    from loop import _dedup_final
    joined = ("### 1. fire away\n**Meaning:** invitation to start asking questions.\n"
              "From this video: \"Okay, fire away.\"\n\n---\n\n"
              "### 2. no can do\n**Meaning:** informal refusal used with friends.\n"
              "From this video: \"No can do, sis.\"")
    # (a) verbatim re-dump -> ""
    assert _dedup_final(joined, joined) == "", "verbatim recap must be dropped"
    # (b) paraphrased per-term recap pruned; new content kept
    extra = ("### Batch 1\n"
             "1. **fire away**: An invitation to start speaking.\n"
             "2. **no can do**: A casual way to refuse.\n"
             "\nNote: you already learned 'take care of' before.\n"
             "Would you like me to stage any of these phrases?")
    out = _dedup_final(joined, extra)
    assert "**fire away**" not in out and "**no can do**" not in out, ("recap lines pruned", out)
    assert "Batch 1" not in out, ("emptied heading pruned", out)
    assert "already learned" in out and "stage any of these" in out, ("new content kept", out)
    print("finish dedup OK -> recap dropped/pruned; novel note + follow-up kept")


def test_recall_surfaces_review_queue():
    """S18 (owner) — recall must know the REVIEW QUEUE: a word staged but not yet committed
    is reported under `in_review_queue` (so the agent never says 'you haven't learned this'
    while it awaits approval); a word in neither graph nor queue gets no such key."""
    from recall import recall
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    try:
        review_io.export_review([{"node": {"key": "flabbergast#flabbergast.v.01",
                                           "term": "flabbergast", "sense_id": "flabbergast.v.01",
                                           "occurrences": [{"sentence": "I was flabbergasted."}]},
                                  "ai_fields": ["ungrounded"], "needs_review": True}],
                                mode="overwrite")
        hit = recall("flabbergast")
        assert not hit["found"], hit                     # not committed to the graph
        q = hit.get("in_review_queue")
        assert q and q[0]["key"].startswith("flabbergast#"), ("queued word must surface", hit)
        assert q[0]["flagged_ungrounded"] is True, q
        assert "in_review_queue" not in recall("zebra crossing"), "unknown word must have no queue key"
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
    print("recall queue OK -> staged-but-uncommitted word surfaced; unknown word clean")


def test_one_sentence_one_sense():
    """S18 (owner) — the SAME source sentence cannot ground TWO senses of one term (a film
    line has ONE meaning; seen live: find out#determine.v.08 + #learn.v.02 both citing 'How
    did you find out?'). Re-staging the same term+sentence under a DIFFERENT sense must SNAP
    onto the existing sense (one queue row); a DIFFERENT sentence keeps its own sense (true
    polysemy, e.g. 'go on')."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    S = "How did you find out?"
    def draft(key, sense, sent, definition):
        return {"node": {"key": key, "term": "find out", "sense_id": sense,
                         "definition": definition, "occurrences": [{"sentence": sent}]}}
    try:
        review_io.export_review([draft("find out#learn.v.02", "learn.v.02", S, "become aware")],
                                mode="append")
        # same sentence, different sense -> snapped onto learn.v.02 (NO second row)
        review_io.export_review([draft("find out#determine.v.08", "determine.v.08", S,
                                       "establish by inquiry")], mode="append")
        keys = [k for k in review_io.load_pending() if k != "_meta"]
        assert keys == ["find out#learn.v.02"], ("same sentence must not fork a 2nd sense", keys)
        # different sentence -> its own sense row stays (true polysemy preserved)
        review_io.export_review([draft("find out#discover.v.01", "discover.v.01",
                                       "We'll find out the truth eventually.", "discover")],
                                mode="append")
        keys = sorted(k for k in review_io.load_pending() if k != "_meta")
        assert keys == ["find out#discover.v.01", "find out#learn.v.02"], keys
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
    print("one-sentence-one-sense OK -> same line snaps to existing sense; real polysemy kept")


def _patch_agent(ai_responses, tool_impl):
    """Monkeypatch the agent loop's AI + tool bindings; returns (loop, counters, restore_fn).
    `ai_responses` = list of raw JSON decision strings popped per call_ai; `tool_impl(name,
    args)` = fake registry. Counts every call_ai so tests can assert LLM-call budgets."""
    import loop
    counters = {"ai": 0, "tools": []}
    real_ai, real_tool, real_key = loop.call_ai, loop.call_tool, loop.config.has_ai_key

    def fake_ai(prompt, system):
        counters["ai"] += 1
        if not ai_responses:
            raise AssertionError("unexpected extra call_ai (LLM budget exceeded in test)")
        return ai_responses.pop(0)

    def fake_tool(name, args):
        counters["tools"].append((name, args))
        return tool_impl(name, args)

    loop.call_ai, loop.call_tool = fake_ai, fake_tool
    loop.config.has_ai_key = lambda: True

    def restore():
        loop.call_ai, loop.call_tool, loop.config.has_ai_key = real_ai, real_tool, real_key
    return loop, counters, restore


def test_explain_ends_turn_no_extra_llm_call():
    """askfix REBASE (S16 frame) — after `explain` returns text, the turn ENDS: no further
    call_ai (the old post-explain decision round let the model chain/repeat), and the scratch
    keeps only a SHORT note (never the full explanation, so a later turn has nothing to copy)."""
    EXPL = "### 1. fire away\n**Meaning:** an invitation to start asking questions."

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "wordnet_lookup":
            return {"senses": []}
        if name == "explain":
            return EXPL
        raise AssertionError(f"unexpected tool {name}")

    loop, counters, restore = _patch_agent(
        ['{"thought":"t","action":{"tool":"explain","args":{"query":"fire away"}}}'], tools)
    try:
        out = loop.run_agent("what does 'fire away' mean?")
    finally:
        restore()
    assert counters["ai"] == 1, ("explain must END the turn — exactly 1 LLM call", counters)
    # content delivered (S19 #2 may re-space field labels, so check content, not exact newlines)
    assert "fire away" in out["answer"] and "an invitation to start asking questions." in out["answer"], out["answer"]
    assert not any(EXPL in s for s in out["scratch"]), ("full explanation must NOT enter scratch", out["scratch"])
    print("explain-ends-turn OK -> 1 LLM call, answer delivered, scratch keeps only a short note")


def test_continuation_deterministic_zero_llm_decisions():
    """askfix REBASE FIX A — 'explain the rest' runs recall -> wordnet_lookup -> explain on the
    REMAINING terms via PYTHON (0 call_ai), so it can never re-explain / copy batch 1. The new
    answer covers exactly the remaining terms and the scratch carries the source file forward."""
    NEW = "### 1. no can do\n**Meaning:** an informal refusal.\n### 2. blend in\n**Meaning:** to look like everyone else."
    OLD_BATCH = "**fire away**: an invitation to start asking questions."
    cbak = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "No can do, sis. You have to blend in. For crying out loud, guys!")

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "wordnet_lookup":
            return {"senses": []}
        if name == "explain":
            assert "no can do" in args["query"] and "blend in" in args["query"], args
            return NEW
        raise AssertionError(f"unexpected tool {name}")

    loop, counters, restore = _patch_agent([], tools)   # NO ai responses -> any call_ai fails
    prior = ["USER QUERY: find phrases",
             f"ATTACHED SOURCE FILE: {SOURCE}",
             "ACTION: explain(\"fire away\")",
             "OBSERVATION: explanation delivered to the learner (shown in full to them; not repeated here).",
             "REMAINING_UNEXPLAINED: no can do, blend in"]
    try:
        out = loop.run_agent("explain the rest", prior_scratch=prior)
    finally:
        restore()
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    assert counters["ai"] == 0, ("continuation must be Python-driven — 0 LLM decisions", counters)
    tools_called = [t for t, _ in counters["tools"]]
    assert tools_called == ["recall", "wordnet_lookup", "explain"], tools_called
    assert "no can do" in out["answer"] and "blend in" in out["answer"], out["answer"]
    assert OLD_BATCH not in out["answer"], ("batch-1 text must not reappear", out["answer"])
    traj = [t["tool"] for t in out["trajectory"]]
    assert traj == ["recall", "wordnet_lookup", "explain"], traj
    print("continuation OK -> deterministic recall->wordnet->explain on remaining, 0 LLM decisions")


def test_recall_miss_note_surfaces_transcript_line():
    """askfix D3 (FIX B) — recall miss on a term that IS in the cached transcript must append
    the anti-lie NOTE with the verbatim cue (live bug: 'crying out loud' -> agent said it was
    not in the script; it is, @ 00:08:05 'For crying out loud, guys!')."""
    cbak = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "Hello there. For crying out loud, guys! Bye now.")

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        raise AssertionError(f"unexpected tool {name}")

    loop, counters, restore = _patch_agent(
        ['{"thought":"check the graph","action":{"tool":"recall","args":{"lemma":"crying out loud"}}}',
         '{"final":"Yes — the script says: \\"For crying out loud, guys!\\""}'], tools)
    try:
        out = loop.run_agent("is 'crying out loud' in the script?", source=SOURCE)
    finally:
        restore()
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    note_lines = [s for s in out["scratch"] if "OBSERVATION" in s and "NOTE:" in s]
    assert note_lines, ("recall-miss NOTE must be appended to the observation", out["scratch"])
    assert 'For crying out loud, guys!' in note_lines[0], ("verbatim cue in the NOTE", note_lines[0])
    assert "GRAPH only" in note_lines[0] and "stage_for_review" in note_lines[0], note_lines[0]
    # live T3 regression: a FOUND word must ALSO get its verbatim cue (the model otherwise
    # invents script lines when asked "is it in the script?" about an already-learned word).
    cbak2 = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "Hello there. For crying out loud, guys! Bye now.")
    try:
        import loop as _loop
        note = _loop._recall_miss_note({"found": True}, {"lemma": "crying out loud"}, [SOURCE])
        assert 'For crying out loud, guys!' in note and "never" in note, note
    finally:
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak2)
    print("recall-miss NOTE OK -> transcript cue surfaced verbatim (miss AND found); no invented script lines")


def test_explain_429_sentinel_keeps_remaining():
    """askfix — explain degrades to an error STRING (not an exception) when the AI call fails
    past all 429 backoff retries. That string must NOT count as a delivered explanation:
    (a) _explain_ok rejects the sentinels; (b) on the deterministic continuation path the
    REMAINING_UNEXPLAINED line survives in scratch, so the learner can retry once quota is back."""
    from loop import _explain_ok
    assert not _explain_ok("(Could not generate an explanation right now: 429 quota)")
    assert not _explain_ok("(No explanation produced.)")
    assert not _explain_ok("")
    assert _explain_ok("### 1. fire away\n**Meaning:** go ahead.")

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "wordnet_lookup":
            return {"senses": []}
        if name == "explain":
            return "(Could not generate an explanation right now: 429 RESOURCE_EXHAUSTED)"
        raise AssertionError(f"unexpected tool {name}")

    loop, counters, restore = _patch_agent(
        ['{"final":"The AI quota is exhausted right now — please try again in a minute."}'], tools)
    prior = ["USER QUERY: find phrases",
             f"ATTACHED SOURCE FILE: {SOURCE}",
             "REMAINING_UNEXPLAINED: no can do, blend in"]
    try:
        out = loop.run_agent("explain the rest", prior_scratch=prior)
    finally:
        restore()
    assert "Could not generate" not in out["answer"], ("error string must not be the answer", out["answer"])
    assert any(s.startswith("REMAINING_UNEXPLAINED:") and "no can do" in s
               for s in out["scratch"]), ("remaining terms must survive a 429 turn", out["scratch"])
    print("429 sentinel OK -> error text rejected; REMAINING preserved for a retry")


def test_b1_fabricated_common_word_sentence_flagged():
    """S19 (B1) — the grounding gate must catch a fabricated sentence built ENTIRELY from common
    words that each appear SOMEWHERE in the film. The old token-overlap check compared against the
    whole space-joined transcript blob, so such a sentence scored ~1.0 and passed ungrounded (an
    invented line then reached Anki). Per-SENTENCE-UNIT overlap closes the hole. The term is NOT a
    real cue (so snap can't rescue it), forcing the check onto the AI's cited sentence."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    # every word of the fabricated line below appears SCATTERED across these lines, but the phrase
    # "call it a day" is not a real cue and no single line contains most of the fabricated words.
    TX = ("Let's start early. Just relax for now. It was a good call. "
          "We took a day off. And then we go. Come home safely.")
    FAB = "Let's just call it a day and go home now."   # all common words, spread across the film
    try:
        import stage_for_review as sfr
        config.AI_API_KEY = "fake"
        en.call_ai = lambda p, s: json.dumps([{"term": "call it a day", "sense_id": "", "confidence": 0.8}])
        cache_transcript(SOURCE, TX)
        out = sfr.stage_for_review(terms=["call it a day"], source=SOURCE,
                                   sentences={"call it a day": FAB})
        ung = [u["term"] for u in out.get("ungrounded", [])]
        assert "call it a day" in ung, ("fabricated common-word sentence must be flagged", out)
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
    print("B1 OK -> fabricated common-word sentence flagged (per-unit overlap, not whole-blob)")


def test_b2_snaps_via_transcript_text_when_no_srt():
    """S19 (B2) — when NO srt is cached (source drift / mine-via-chat), _ground_sentence used to
    give up and keep the AI's (possibly hallucinated) sentence. Now it falls back to the cached
    transcript TEXT split into sentence units and snaps the term to its REAL line via ground_line.
    No srt -> no fabricated timestamp (correct); only ungrounded->grounded for a genuine line."""
    pbak = _stash(review_io.PENDING_PATH)
    gbak = _stash(GRAPH_PATH)
    cbak = _stash(_transcript_cache_path(SOURCE))
    sbak = _stash(_srtpath_cache_path(SOURCE))    # B2 = NO srt: clear any stale sidecar first
    full = "I've tried to make it work, really. That's no reason to get a divorce."
    try:
        import stage_for_review as sfr
        config.AI_API_KEY = "fake"
        en.call_ai = lambda p, s: json.dumps([{"term": "make it work", "sense_id": "", "confidence": 0.8}])
        cache_transcript(SOURCE, full)        # NOTE: no srt_path -> load_cached_srt returns ""
        out = sfr.stage_for_review(
            terms=["make it work"], source=SOURCE,
            sentences={"make it work": "Can't you do something like make it work?"})  # hallucinated frame
        pend = review_io.load_pending()
        occ = next(v["node"]["occurrences"][0] for k, v in pend.items()
                   if k != "_meta" and isinstance(v, dict))
        assert occ["sentence"] == "I've tried to make it work, really.", ("snapped via transcript text", occ)
        assert "make it work" not in [u["term"] for u in out.get("ungrounded", [])], out
        assert not occ.get("start") and not occ.get("start_sec"), ("no srt -> no timestamp", occ)
    finally:
        for path in (review_io.PENDING_PATH, GRAPH_PATH, _transcript_cache_path(SOURCE),
                     _srtpath_cache_path(SOURCE)):
            if os.path.exists(path):
                os.remove(path)
        _restore(review_io.PENDING_PATH, pbak)
        _restore(GRAPH_PATH, gbak)
        _restore(_transcript_cache_path(SOURCE), cbak)
        _restore(_srtpath_cache_path(SOURCE), sbak)
    print("B2 OK -> transcript-text snap when no srt cached; no fabricated timestamp")


def test_a1_save_to_queue_stages_not_explains():
    """S19 (A1) — a SAVE request ('ghi các từ còn lại vào queue') must STAGE the remaining terms,
    never be hijacked into re-explaining. The old continuation regex matched 'còn lại' and force-
    ran explain. Now Python routes it to stage_for_review deterministically (0 LLM decisions)."""
    cbak = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "No can do, sis. You have to blend in.")
    staged = {"newly_staged": ["no can do", "blend in"], "already_present": [], "ungrounded": []}

    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "stage_for_review":
            assert "no can do" in args["terms"] and "blend in" in args["terms"], args
            return staged
        raise AssertionError(f"unexpected tool {name} (explain must NOT run on a save request)")

    loop, counters, restore = _patch_agent([], tools)    # NO ai responses -> any call_ai fails
    prior = ["USER QUERY: find phrases",
             f"ATTACHED SOURCE FILE: {SOURCE}",
             "REMAINING_UNEXPLAINED: no can do, blend in"]
    try:
        out = loop.run_agent("ghi các từ còn lại vào queue", prior_scratch=prior)
    finally:
        restore()
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    assert counters["ai"] == 0, ("save request must be Python-driven — 0 LLM decisions", counters)
    tools_called = [t for t, _ in counters["tools"]]
    assert "stage_for_review" in tools_called, ("must STAGE the remaining terms", tools_called)
    assert "explain" not in tools_called, ("must NOT explain on a save request", tools_called)
    assert "review queue" in out["answer"].lower(), out["answer"]
    print("A1 OK -> 'save the rest to queue' stages remaining terms, never re-explains")


def test_a2_exhausted_continuation_no_reexplain_and_marker_reset():
    """S19 (A2) — two guarantees against the infinite 'last answer' loop:
      (1) finishing the LAST remaining terms writes an EMPTY REMAINING_UNEXPLAINED marker (the old
          code left the stale batch in scratch, so 'explain the rest' re-explained it forever);
      (2) a continuation request with nothing left returns a deterministic 'all caught up' reply
          with ZERO explain / LLM calls."""
    cbak = _stash(_transcript_cache_path(SOURCE))
    cache_transcript(SOURCE, "No can do, sis. You have to blend in.")

    def tools_explain(name, args):
        if name == "recall":
            return {"found": False}
        if name == "wordnet_lookup":
            return {"senses": []}
        if name == "explain":
            return "### 1. no can do\n**Meaning:** an informal refusal.\n### 2. blend in\n**Meaning:** to fit in."
        raise AssertionError(f"unexpected tool {name}")

    # turn 1: explain the LAST two remaining -> marker must reset to empty
    loop, c1, restore = _patch_agent([], tools_explain)
    prior = ["USER QUERY: find phrases",
             f"ATTACHED SOURCE FILE: {SOURCE}",
             "REMAINING_UNEXPLAINED: no can do, blend in"]
    try:
        out1 = loop.run_agent("explain the rest", prior_scratch=prior)
    finally:
        restore()
    rem_lines = [s for s in out1["scratch"] if s.startswith("REMAINING_UNEXPLAINED:")]
    assert rem_lines, out1["scratch"]
    assert rem_lines[-1].strip() == "REMAINING_UNEXPLAINED:", ("marker must reset to EMPTY", rem_lines)

    # turn 2: another 'explain the rest' with nothing left -> deterministic all-done, 0 calls
    def tools_none(name, args):
        raise AssertionError(f"unexpected tool {name} (nothing left to explain)")

    loop, c2, restore = _patch_agent([], tools_none)
    try:
        out2 = loop.run_agent("explain the rest", prior_scratch=out1["scratch"])
    finally:
        restore()
        p = _transcript_cache_path(SOURCE)
        if os.path.exists(p):
            os.remove(p)
        _restore(p, cbak)
    assert c2["ai"] == 0 and not c2["tools"], ("exhausted continuation must call nothing", c2)
    assert "caught up" in out2["answer"].lower(), out2["answer"]
    print("A2 OK -> last batch resets REMAINING marker; exhausted continuation is a 0-call all-done")


def test_summary_block_surfaces_cross_graph_links():
    """S19 (owner) + BUG-2: a discovered word that connects (synonym/is_a/…) to a word ALREADY in
    the learner's graph is FLAGGED in the summary's 🔗 block (0 AI). BUG-2 upgrades the display: it
    NAMES the relation type + the existing node's meaning (was a bare `X ↔ Y`, which hid a
    sense-mismatch), by reading the REAL edge off the graph, and DROPS antonym links (WordNet
    matches by lemma not sense, so an antonym edge on a same-spelled different sense — the owner's
    'turn out'/'turn in' case — is a false link). An exact repeat is NOT double-reported."""
    from loop import _summary_block
    from schema import PersonalGraph, Node, Edge
    from _common import save_graph
    gbak = _stash(GRAPH_PATH)
    try:
        g = PersonalGraph()
        # existing nodes; each carries an edge pointing back at the newly-discovered term.
        g.upsert(Node(key="reduce#reduce.v.01", term="reduce", sense_id="reduce.v.01",
                      definition="make smaller in amount",
                      edges=[Edge(type="synonym", target="cut", source="wordnet")]))
        g.upsert(Node(key="tusk#tusk.n.01", term="tusk", sense_id="tusk.n.01",
                      definition="a long pointed tooth",
                      edges=[Edge(type="is_a", target="trunk", source="wordnet")]))
        # BUG-2: an ANTONYM edge on a same-lemma DIFFERENT sense must be filtered out (noise).
        g.upsert(Node(key="turn in#turn_in.v.02", term="turn in", sense_id="turn_in.v.02",
                      definition="go to bed",
                      edges=[Edge(type="antonym", target="turn out", source="wordnet")]))
        save_graph(g, GRAPH_PATH)

        cands = [{"term": "cut"}, {"term": "trunk"}, {"term": "turn out"}]
        out = _summary_block(
            cands, explained_terms=["cut", "trunk", "turn out"], known_terms={},
            related_links={"cut": {"reduce#reduce.v.01"}, "trunk": {"tusk#tusk.n.01"},
                           "turn out": {"turn in#turn_in.v.02"}})
        assert "🔗 Connects to your graph" in out, out
        # relation type + meaning are now explicit
        assert "**cut** —synonym→ **reduce**" in out, out
        assert "make smaller in amount" in out, out
        assert "**trunk** —is_a→ **tusk**" in out, out
        # the antonym / sense-mismatch link is DROPPED, not shown
        assert "turn in" not in out and "antonym" not in out, out
        assert "relate to" in out, out                   # the on-demand nudge

        out2 = _summary_block(cands, explained_terms=["cut"],
                              known_terms={"cut": "your collection"}, related_links={})
        assert "🔗" not in out2, out2
    finally:
        _restore(GRAPH_PATH, gbak)
    print("cross-graph links OK -> relation type + meaning named; antonym sense-mismatch dropped")


def test_open6i_strip_be_prefix_grounds():
    """S19 OPEN-6i: the extractor emits copula-headed idioms ('be all over the place', 'be the
    happs') that the film never says with 'be'. As a LAST resort (after every tier fails),
    ground_line strips a leading 'be' and retries -> the real film line grounds. Guarded: a
    2-token 'be X' is NOT stripped, so a bare remainder can't match a line loosely."""
    from _common import ground_line
    lines = ["They're all over the place.", "What's the happs?"]
    assert ground_line("be all over the place", lines) == "They're all over the place."
    assert ground_line("be the happs", lines) == "What's the happs?"
    assert ground_line("be up", ["Look up there.", "They're all over the place."]) is None
    print("OPEN-6i OK -> leading 'be' stripped as a last resort; short 'be X' not over-matched")


def test_open1_find_intent_not_swallowed_by_continuation():
    """S19 OPEN-1: 'tìm câu cho các cụm sau: …' holds the NOUN 'các cụm' but is a FIND request,
    not 'explain the rest'. The old continuation regex listed 'các cụm'/'more phrases', so with an
    exhausted REMAINING marker it fired the 0-call 'all caught up' reply. The find-intent guard
    must skip continuation and route to the LLM instead."""
    def tools(name, args):
        if name == "recall":
            return {"found": False}
        if name == "wordnet_lookup":
            return {"senses": []}
        if name == "explain":
            return "### be all over the place\n**From this video:** They're all over the place."
        return {}
    prior = ["USER QUERY: find phrases", "REMAINING_UNEXPLAINED:"]   # empty = old code -> caught up
    loop, c, restore = _patch_agent(
        ['{"thought":"t","action":{"tool":"explain","args":{"query":"be all over the place"}}}'],
        tools)
    try:
        out = loop.run_agent("tìm câu cho các cụm sau: be all over the place, have a say",
                             prior_scratch=prior)
    finally:
        restore()
    assert "caught up" not in out["answer"].lower(), ("find-intent must NOT hit all-caught-up", out["answer"])
    assert c["ai"] >= 1, ("find-intent must reach the LLM, not the 0-call continuation path", c)
    print("OPEN-1 OK -> find-intent skips continuation regex, routes to LLM (not 'all caught up')")


def test_expand_to_neighbor_nearest_and_robust():
    """S19 OPEN-7: audio clip widens to the PREVIOUS cue's start + NEXT cue's end, and now
    snaps to the NEAREST cue boundary so a few-ms drift no longer BAILS the expansion (the
    old exact <0.05s match left the bare cue + 0.2s pad -> the last word got clipped). A
    wildly-off timestamp still refuses to snap onto an unrelated cue."""
    from loop import _expand_to_neighbor
    segs = [{"start_sec": 0.0, "end_sec": 2.0},
            {"start_sec": 2.0, "end_sec": 4.0},
            {"start_sec": 4.0, "end_sec": 6.0}]
    assert _expand_to_neighbor(segs, 2.0, 4.0) == (0.0, 6.0)      # exact: prev.start .. next.end
    assert _expand_to_neighbor(segs, 2.03, 3.98) == (0.0, 6.0)    # ms drift: still expands
    assert _expand_to_neighbor(segs, 0.0, 2.0) == (0.0, 4.0)      # first cue: no prev, end expands
    assert _expand_to_neighbor(segs, 100.0, 102.0) == (100.0, 102.0)  # far off: unchanged
    print("OPEN-7 expand_to_neighbor OK -> nearest-cue, ms-drift robust, sanity-bounded")


def test_rederive_on_sense_change_refreshes_and_rekeys():
    """S19 OPEN-8: changing a word's sense must refresh every field DERIVED from the synset
    (edges -> card Synonyms/Related, definition, pos) and rekey (term#sense_id), else the card
    keeps the old meaning (the 'put down' bug). ConceptNet edges + learner-typed fields survive."""
    import app
    senses = (app.call_tool("wordnet_lookup", {"term": "dog"}).get("senses") or [])
    assert len(senses) >= 2, "need two WordNet senses for the test"
    old_s, new_s = senses[0], senses[1]
    cn_edge = {"type": "is_a", "target": "life-context-thing", "source": "conceptnet"}
    node = {"key": f"dog#{old_s['sense_id']}", "term": "dog", "sense_id": new_s["sense_id"],
            "definition": old_s["definition"], "pos": old_s.get("pos"),
            "edges": list(old_s["edges"]) + [cn_edge],
            "mnemonic": "old hook", "collocations": ["old colloc"],
            "source_map": {"mnemonic": "ai", "collocations": "ai"}}
    out = app._rederive_on_sense_change(node, new_s["sense_id"])
    assert out["key"] == f"dog#{new_s['sense_id']}", out["key"]
    assert out["definition"] == new_s["definition"]
    assert cn_edge in out["edges"], "conceptnet edge must survive"
    wn_now = {(e["type"], e["target"]) for e in out["edges"] if e.get("source") == "wordnet"}
    assert wn_now == {(e["type"], e["target"]) for e in new_s["edges"]}, "WordNet edges swapped"
    assert out["mnemonic"] is None and out["collocations"] == [], "stale AI fields cleared"
    print("OPEN-8 rederive OK -> edges/def follow new sense, rekeyed, stale mnemonic cleared")


if __name__ == "__main__":
    test_real_line_grounded_fabricated_flagged()
    test_no_transcript_degrades()
    test_restage_updates_ungrounded_sentence()
    test_1e_snaps_sentence_to_real_srt_line()
    test_materials_for_verbatim_wins_and_absent_term_dropped()
    test_ground_line_verbatim_inflection_and_no_false_substring()
    test_ask_fabrication_guard_replaces_bad_from_this_video_quotes()
    test_guard_uses_recall_lines_on_followup_turn()
    test_chat_stage_carries_timestamp()
    test_infolog_chronological_with_definition()
    test_summary_block()
    test_guard_line_local_does_not_touch_notes()
    test_renumber_terms_across_batches()
    test_finish_dedup_drops_recap_keeps_followup()
    test_recall_surfaces_review_queue()
    test_one_sentence_one_sense()
    test_explain_ends_turn_no_extra_llm_call()
    test_continuation_deterministic_zero_llm_decisions()
    test_recall_miss_note_surfaces_transcript_line()
    test_explain_429_sentinel_keeps_remaining()
    test_b1_fabricated_common_word_sentence_flagged()
    test_b2_snaps_via_transcript_text_when_no_srt()
    test_a1_save_to_queue_stages_not_explains()
    test_a2_exhausted_continuation_no_reexplain_and_marker_reset()
    test_format_explain_fields_ungluing()
    test_a1_save_all_stages_full_found_set()
    test_transcript_hint_returns_real_sentence_not_blob()
    test_summary_block_surfaces_cross_graph_links()
    test_open6i_strip_be_prefix_grounds()
    test_open1_find_intent_not_swallowed_by_continuation()
    test_expand_to_neighbor_nearest_and_robust()
    test_rederive_on_sense_change_refreshes_and_rekeys()
    print("OK")
