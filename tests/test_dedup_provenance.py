"""Offline tests for the S9 dedup / provenance / recency work (no key, no network):

  1. lemmatize_term       — deterministic WordNet-gated lemma (single word + WN phrase only).
  2. extract dedup+surface— inflected variants collapse by lemma; original form kept as surface.
  3. extract chunking     — a long script is sampled window-by-window (even coverage).
  4. _collect_repeats     — every sighting stored with its ORIGINAL surface (lemma-aware).
  5. Occurrence.surface   — schema field roundtrips.
  6. build_render_graph   — latest-session nodes get the gold ring + 🆕 marker.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy"),
          os.path.join(ROOT, "agent")):
    sys.path.insert(0, p)

import config
import extract_vocab as ev


def test_lemmatize_term():
    # single word: known inflection -> base; unknown -> unchanged
    assert ev.lemmatize_term("emissions") == "emission"
    assert ev.lemmatize_term("Paris") == "paris"          # not a WN inflection -> just lowered
    # WordNet phrasal verbs -> base form (gate passes)
    assert ev.lemmatize_term("figured out") == "figure out"
    assert ev.lemmatize_term("ran into") == "run into"
    assert ev.lemmatize_term("gave up") == "give up"
    # collocations / proper-noun phrases -> UNCHANGED (gate rejects unsafe lemmas)
    assert ev.lemmatize_term("running shoes") == "running shoes"
    assert ev.lemmatize_term("rising costs") == "rising costs"
    assert ev.lemmatize_term("United States") == "united states"
    print("lemmatize_term OK")


def test_extract_dedup_and_surface():
    """Two inflected variants of one lemma collapse to ONE candidate; surface keeps the
    original form so source data is not lost."""
    config.AI_API_KEY = "fake"
    payload = (
        '[{"term":"emissions","sentence":"We cut emissions today.","tag":"Word"},'
        ' {"term":"emission","sentence":"One emission source.","tag":"Word"},'
        ' {"term":"figured out","sentence":"I figured out the plan.","surface":"figured out","tag":"Phrasal Verb"},'
        # collocation: the AI returns the base form; WordNet has no synset for it, so the
        # AI-lemma must STAND (not be reverted) and surface keeps the conjugated form.
        ' {"term":"make a decision","sentence":"They made a decision fast.","surface":"made a decision","tag":"Collocation"}]'
    )
    ev.call_ai = lambda prompt, sysp, model=None: payload
    out = ev.extract_vocab(["We cut emissions today.", "One emission source.",
                            "I figured out the plan.", "They made a decision fast."])
    terms = [c["term"] for c in out]
    assert terms == ["emission", "figure out", "make a decision"], terms   # deduped + lemmatized
    by_term = {c["term"]: c for c in out}
    assert by_term["emission"]["surface"] == "emissions"       # original form preserved
    assert by_term["figure out"]["surface"] == "figured out"
    assert by_term["make a decision"]["surface"] == "made a decision"   # AI-lemma stands
    print("extract dedup+surface OK:", terms)


def test_extract_chunking_covers_all_windows():
    """A >EXTRACT_CHUNK_LINES transcript is split into windows and gathered from EACH, so a
    term living only in the LAST window is still surfaced (the single-pass bias the OPEN
    item flagged)."""
    config.AI_API_KEY = "fake"
    # 600 lines, each with a unique grounded token wNNN -> 3 windows of 200 (cap 6).
    lines = [f"line w{i} appears here" for i in range(600)]
    transcript = "\n".join(lines)
    assert len(ev._split_into_chunks(transcript)) > 1          # chunking engaged

    calls = []
    def fake(prompt, sysp, model=None):
        calls.append(prompt)
        import re
        toks = re.findall(r"w\d+", prompt.split("TRANSCRIPT:")[-1])
        tok = toks[0] if toks else "w0"                        # first token IN THIS window
        return f'[{{"term":"{tok}","sentence":"line {tok} appears here","surface":"{tok}","tag":"Word"}}]'
    ev.call_ai = fake

    out = ev.extract_vocab(transcript)
    got = {c["term"] for c in out}
    assert len(calls) >= 3, len(calls)                         # one call per window (+maybe fix)
    assert "w0" in got                                         # first window covered
    assert any(int(t[1:]) >= 400 for t in got), got            # a LATE-window term covered too
    print(f"chunking OK: {len(calls)} calls, terms={sorted(got)}")


def test_collect_repeats_keeps_all_surfaces():
    import loop
    node = {"term": "reduce",
            "occurrences": [{"source": "ep1", "sentence": "We must reduce it.",
                             "surface": "reduce", "added_at": "2026-06-30"}]}
    segments = [
        {"text": "We must reduce it.", "start": "00:00:01", "end": "00:00:02"},   # primary (seen)
        {"text": "They reduced the cost.", "start": "00:00:03", "end": "00:00:04"},
        {"text": "She reduces it daily.", "start": "00:00:05", "end": "00:00:06"},
        {"text": "Totally unrelated line.", "start": "00:00:07", "end": "00:00:08"},
    ]
    loop._collect_repeats(node, segments)
    surfaces = {o["surface"] for o in node["occurrences"]}
    assert surfaces == {"reduce", "reduced", "reduces"}, surfaces   # inflections captured
    assert all("unrelated" not in o["sentence"] for o in node["occurrences"])
    # multi-word: verbatim only (no token-split lemmatization)
    mw = {"term": "figure out",
          "occurrences": [{"source": "ep1", "sentence": "I figured out it.",
                           "surface": "figured out", "added_at": "x"}]}
    segs2 = [{"text": "I figured out it.", "start": "", "end": ""},
             {"text": "We figured out later.", "start": "", "end": ""}]
    loop._collect_repeats(mw, segs2)
    assert {o["surface"] for o in mw["occurrences"]} == {"figured out"}
    print("collect_repeats OK:", sorted(surfaces))


def test_occurrence_surface_roundtrips():
    from schema import Occurrence, PersonalGraph, Node
    occ = Occurrence(source="ep1", sentence="I figured out it.", surface="figured out")
    assert occ.surface == "figured out"
    g = PersonalGraph()
    g.upsert(Node(key="figure out#x", term="figure out", occurrences=[occ]))
    import json
    g2 = PersonalGraph.model_validate_json(g.model_dump_json())
    assert g2.nodes["figure out#x"].occurrences[0].surface == "figured out"
    print("Occurrence.surface roundtrip OK")


def test_recent_nodes_get_gold_ring():
    from build_render_graph import build_render_graph, RECENT_BORDER
    nodes = [
        {"key": "reduce#reduce.v.01", "term": "reduce", "edges": [], "occurrences": []},
        {"key": "gas#gas.n.01", "term": "gas", "edges": [], "occurrences": []},
    ]
    html_path = build_render_graph(nodes, run_id="test_recent",
                                   recent=["gas#gas.n.01"])      # only 'gas' is from this session
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    assert RECENT_BORDER in html, "gold ring colour missing"
    assert "added this session" in html or "added in the latest session" in html
    print("recent gold-ring OK ->", os.path.basename(html_path))


def test_approve_all_marks_every_row():
    """NON-EXCEL 'Approve All': set status='approved' on every review row (df-session).

    Mirrors app.py's sidebar Approve All (df['status'] = 'approved'), reading rows from
    pending_to_rows — the source the in-app st.data_editor is seeded from.
    """
    import tempfile
    import pandas as pd
    import review_io
    drafts = [
        {"node": {"term": "reduce", "key": "reduce#reduce.v.01"},
         "confidence": 0.9, "needs_review": False, "ai_fields": [], "surface": "reduce"},
        {"node": {"term": "emission", "key": "emission#emission.n.01"},
         "confidence": 0.8, "needs_review": True, "ai_fields": ["mnemonic"], "surface": "emissions"},
    ]
    d = tempfile.mkdtemp()
    pend = os.path.join(d, "pending.json")
    review_io.export_review(drafts, pending_path=pend)
    rows = review_io.pending_to_rows(review_io.load_pending(pend))
    assert len(rows) == 2, rows
    df = pd.DataFrame(rows, columns=review_io.COLUMNS)
    df["status"] = "approved"                       # <-- the app's Approve All
    assert list(df["status"].str.lower()) == ["approved", "approved"], df["status"].tolist()
    print("approve_all (df-session) OK:", len(df), "rows")


def test_locate_timestamp_stopword_safe():
    """S14 T9: tier-3 token fallback anchors on the longest CONTENT token; a surface made
    only of stopwords returns ("","") instead of matching the first "be"/"a" line."""
    import loop
    segments = [
        {"text": "To be or not to be, that is a question.", "start": "00:00:01", "end": "00:00:02"},
        {"text": "She finally figured the whole thing.", "start": "00:01:00", "end": "00:01:02"},
    ]
    # (a) content token "figured" -> the segment that contains it, not the stopword line
    assert loop._locate_timestamp(segments, "figured it out", "") == ("00:01:00", "00:01:02")
    # (b) all-stopword surface -> no tier-3 guess
    assert loop._locate_timestamp(segments, "be in", "") == ("", "")
    # tier 1 (verbatim) still wins
    assert loop._locate_timestamp(segments, "figured the whole", "") == ("00:01:00", "00:01:02")
    print("locate_timestamp stopword-safe OK")


if __name__ == "__main__":
    test_lemmatize_term()
    test_extract_dedup_and_surface()
    test_extract_chunking_covers_all_windows()
    test_collect_repeats_keeps_all_surfaces()
    test_occurrence_surface_roundtrips()
    test_recent_nodes_get_gold_ring()
    test_approve_all_marks_every_row()
    test_locate_timestamp_stopword_safe()
    print("OK")
