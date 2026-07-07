"""
Milestone integration: Path A (query) end-to-end with a mocked AI call.

  recall (miss) -> wordnet_lookup -> enrich(AI) -> upsert(grow) -> save/load
  -> build_render_graph (HTML) + make_anki (.apkg)

Proves the deterministic spine works without any network/key. (Path B just prepends
ingest_transcript -> extract_vocab, both covered by their own tests.)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import enrich as en
from recall import recall
from wordnet_lookup import wordnet_lookup
from build_render_graph import build_render_graph
from make_anki import make_anki
from schema import PersonalGraph, Node

GRAPH = "data/_test_pipeline_graph.json"


def test_path_a_query():
    config.AI_API_KEY = "fake"
    term, sentence = "gas", "reduce carbon gas emissions"

    # 1. recall first (fresh graph -> miss)
    if os.path.exists(GRAPH):
        os.remove(GRAPH)
    assert recall(term, graph_path=GRAPH)["found"] is False

    # 2. wordnet_lookup (deterministic senses)
    senses = wordnet_lookup(term)["senses"]
    assert senses

    # 3. enrich (mock the ONE AI call) -> draft Node
    en.call_ai = lambda p, s: json.dumps([{
        "term": term, "sense_id": "gas.n.02",
        "collocations": ["natural gas"], "mnemonic": "gas=air", "pattern": "reduce <gas>",
        "confidence": 0.9}])
    drafts = en.enrich([{"term": term, "sentence": sentence, "senses": senses, "source": "demo"}])
    node = Node(**drafts[0]["node"])
    assert node.source_map["sense_id"] == "ai"
    assert all(e.source == "wordnet" for e in node.edges)

    # 4. upsert into the (growing) graph + persist
    g = PersonalGraph.load(GRAPH)
    g.upsert(node)
    g.save(GRAPH)
    g2 = PersonalGraph.load(GRAPH)
    assert g2.recall(term)["as_main_node"] is not None      # now remembered

    # 5. render + deck
    html = build_render_graph([d["node"] for d in drafts], run_id="test_pipeline")
    deck = make_anki([{"node": drafts[0]["node"], "surface": "gas"}],
                     deck_name="Pipeline", run_id="test_pipeline")
    assert os.path.exists(html) and os.path.exists(deck["apkg"]) and deck["n_cards"] >= 2

    os.remove(GRAPH)
    print(f"Path A OK -> node {node.key}, {deck['n_cards']} cards, html {os.path.basename(html)}")


def test_loop_tool_call_cap():
    """S14 T4: a model that ALWAYS returns an action must execute at most max_tool_calls tools."""
    config.AI_API_KEY = "fake"
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import loop

    calls = {"n": 0}

    def fake_call_tool(tool, args):
        calls["n"] += 1
        return {"found": False}

    orig_ai, orig_tool = loop.call_ai, loop.call_tool
    loop.call_ai = lambda prompt, system: json.dumps(
        {"thought": "again", "action": {"tool": "recall", "args": {"lemma": "x"}}})
    loop.call_tool = fake_call_tool
    try:
        out = loop.run_agent("test", max_tool_calls=8)
    finally:
        loop.call_ai, loop.call_tool = orig_ai, orig_tool
    assert calls["n"] <= 8, f"cap says 8 but {calls['n']} tools executed"
    assert isinstance(out.get("answer", ""), str)
    print(f"loop cap OK -> {calls['n']} tool calls (<= 8)")


def test_observe_recall_surfaces_provenance():
    """S16 T-A3: _observe on a recall hit must surface the meaning source + film name(s) so
    the agent can answer 'who defined X? which film?' — the generic dict compaction drops them."""
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import loop

    result = {
        "found": True,
        "as_main_node": {
            "term": "fed up", "sense_id": "fed_up.a.01",
            "source_map": {"definition": "wordnet"},
            "occurrences": [
                {"source": "Charade", "sentence": "I'm fed up with waiting."},
                {"source": "Charade", "sentence": "still fed up?"},
                {"source": "Notting Hill", "sentence": "fed up again"},
            ],
        },
        "as_related": [], "in_sentences": [], "in_collocations": [],
    }
    obs = loop._observe(result)
    assert "wordnet" in obs, obs                 # definition_source
    assert "Charade" in obs and "Notting Hill" in obs, obs   # distinct film sources
    assert "n_occurrences" in obs, obs
    assert "ALREADY A LEARNED WORD" in obs, obs  # agent must TELL the learner (TH1)

    # TH2 (S16+ ext): not a learned node, but SEEN as a related word / in a sentence.
    # as_related holds node KEYS (schema.recall appends keys, not dicts).
    related_only = {
        "found": False, "as_main_node": None,
        "as_related": ["tusk#tusk.n.01", "ivory#ivory.n.01"],
        "in_sentences": [{"node": "tusk#tusk.n.01", "sentence": "the tusk of an elephant"}],
        "in_collocations": [],
    }
    obs2 = loop._observe(related_only)
    assert "SEEN BEFORE" in obs2, obs2
    assert "tusk" in obs2, obs2                  # where it appeared
    # completely unknown word -> no provenance tail at all
    obs3 = loop._observe({"found": False, "as_main_node": None,
                          "as_related": [], "in_sentences": [], "in_collocations": []})
    assert "SEEN BEFORE" not in obs3 and "ALREADY" not in obs3, obs3
    print("observe recall provenance OK -> main/related/none all correct")


def test_ingest_vtt_txt_md():
    """S16 T8: ingest_transcript accepts .vtt (timed), .txt and .md (untimed). .vtt keeps
    HH:MM:SS timings; .txt/.md yield segments with empty start/end but real text; a .txt
    then runs through extract_vocab -> enrich -> stage with the AI mocked (offline)."""
    import tempfile
    from ingest_transcript import ingest_transcript
    d = tempfile.mkdtemp()

    vtt = os.path.join(d, "clip.vtt")
    with open(vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nWe must reduce carbon emissions.\n\n"
                "00:00:04.500 --> 00:00:07.000\nThe deadline is tight.\n")
    vsegs = ingest_transcript(vtt)["segments"]
    assert len(vsegs) == 2, vsegs
    assert vsegs[0]["start"] == "00:00:01" and vsegs[0]["end"] == "00:00:04", vsegs[0]
    assert "reduce carbon emissions" in vsegs[0]["text"], vsegs[0]

    txt = os.path.join(d, "notes.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("We must reduce carbon emissions.\n\nThe deadline is tight.\n")
    tsegs = ingest_transcript(txt)["segments"]
    assert len(tsegs) == 2, tsegs
    assert tsegs[0]["start"] == "" and tsegs[0]["end"] == "", tsegs[0]
    assert tsegs[0]["text"] == "We must reduce carbon emissions.", tsegs[0]

    md = os.path.join(d, "notes.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Heading\n\n- We **must** reduce [carbon](http://x) emissions.\n")
    msegs = ingest_transcript(md)["segments"]
    assert any(s["text"] == "Heading" for s in msegs), msegs
    assert any("must reduce carbon emissions" in s["text"] for s in msegs), msegs

    # the untimed .txt result feeds the mining spine cleanly: extract_vocab consumes the
    # ingest dict via _as_text, and an empty start/end does not break timestamp location.
    from extract_vocab import _as_text
    ing_txt = ingest_transcript(txt)
    assert "reduce carbon emissions" in _as_text(ing_txt)
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    from loop import _locate_timestamp
    # untimed segments -> no crash, returns empty timings (not a wrong guess)
    assert _locate_timestamp(tsegs, "reduce", "We must reduce carbon emissions.") == ("", "")
    print("ingest vtt/txt/md OK -> vtt timed, txt/md untimed, downstream-consumable")


def test_batch_comma_terms():
    """S17-5.1a: the agent sometimes packs many terms into ONE comma-separated call
    ("make it work, go on, …") — recall/wordnet_lookup used to look up the whole string
    and silently return found=False. Now they split per term; explain grounds itself
    via real recall when the agent passes no usable context (labels = evidence, not vibes)."""
    tmp_graph = "data/_test_batch_graph.json"
    if os.path.exists(tmp_graph):
        os.remove(tmp_graph)
    g = PersonalGraph.load(tmp_graph)
    g.upsert(Node(term="go on", key="go on#go_on.v.01", word_type="phrasal_verb",
                  definition="continue", source_map={"definition": "wordnet"},
                  occurrences=[{"source": "Charade.srt", "sentence": "Go on, tell me."}]))
    g.save(tmp_graph)
    try:
        # recall: comma-list -> per-term results, no more silent found=False
        out = recall("make it work, go on", graph_path=tmp_graph)
        assert out["found"] is True, out                      # 'go on' IS in the graph
        assert set(out["batch"]) == {"make it work", "go on"}, out
        assert out["batch"]["go on"]["found"] is True
        assert out["batch"]["make it work"]["found"] is False
        assert "ONE lemma" in out["note"]

        # wordnet_lookup: same split; per-term senses returned
        w = wordnet_lookup("gas, flyblowxyz")
        assert w["found"] is True and set(w["batch"]) == {"gas", "flyblowxyz"}, w
        assert w["batch"]["gas"]["senses"] and not w["batch"]["flyblowxyz"]["senses"]

        # _observe surfaces EACH term (incl. ALREADY-A-LEARNED-WORD for the known one)
        sys.path.insert(0, os.path.join(ROOT, "agent"))
        import loop
        obs = loop._observe(out)
        assert "BATCH of 2 terms" in obs and "[go on]" in obs and "[make it work]" in obs, obs
        assert "ALREADY A LEARNED WORD" in obs, obs

        # explain (agent path, NO context): self-grounds via real recall per term — the
        # prompt sent to the AI must contain the learner's real prior occurrence.
        import explain as ex
        config.AI_API_KEY = "fake"
        seen = {}
        ex.call_ai = lambda prompt, system: (seen.__setitem__("prompt", prompt)
                                             or '{"explanation": "ok"}')
        from _common import GRAPH_PATH as _real_gp
        import recall as rc
        real_recall = rc.recall
        rc.recall = lambda lemma, graph_path=_real_gp: real_recall(lemma, tmp_graph)
        try:
            ans = ex.explain("make it work, go on")
            assert ans == "ok"
            assert "Charade.srt" in seen["prompt"], seen["prompt"][:300]
            assert "Go on, tell me." in seen["prompt"]
        finally:
            rc.recall = real_recall
    finally:
        if os.path.exists(tmp_graph):
            os.remove(tmp_graph)
    print("batch comma terms OK -> recall/wordnet split per term; explain self-grounded")


def test_anki_phase2_cards_and_extra():
    """S18 Phase 2: make_anki builds the NEW Definition note type (additive), folds the rich
    info + other occurrences into a <details>, and always shows a Source line. The 3 legacy
    model_ids are UNCHANGED (history-preserving invariant)."""
    import zipfile
    import json as _json
    import sqlite3
    import tempfile
    from make_anki import make_anki, _BASIC_ID, _CLOZE_ID, _DICT_ID, _DEF_ID
    # invariant: legacy ids fixed; the new Definition id is distinct from all three
    assert (_BASIC_ID, _CLOZE_ID, _DICT_ID) == (1607392319, 1607392320, 1607392321)
    assert _DEF_ID not in (_BASIC_ID, _CLOZE_ID, _DICT_ID)

    node = {"key": "reduce#reduce.v.01", "term": "reduce", "pos": "verb",
            "definition": "make smaller", "collocations": ["reduce costs"],
            "tags": ["finance"],
            "edges": [{"type": "synonym", "target": "decrease#decrease.v.01", "source": "wordnet"},
                      {"type": "is_a", "target": "change#change.v.01", "source": "wordnet"}],
            "occurrences": [
                {"source": "Charade.srt", "sentence": "We must reduce costs.",
                 "start": "00:00:10", "end": "00:00:12"},
                {"source": "Wall_Street.srt", "sentence": "They reduce the workforce.",
                 "start": "00:05:00", "end": "00:05:03"}]}
    deck = make_anki([node], deck_name="Demo", run_id="phase2_test")
    assert os.path.exists(deck["apkg"]), deck
    # Basic + Definition + Cloze (no audio -> no Dictation)
    assert deck["n_cards"] == 3, deck

    out = tempfile.mkdtemp()
    z = zipfile.ZipFile(deck["apkg"])
    z.extract("collection.anki2", out)
    db = sqlite3.connect(os.path.join(out, "collection.anki2"))
    names = {m["name"] for m in _json.loads(db.execute("select models from col").fetchone()[0]).values()}
    assert "VocaSync Definition" in names, names
    flds = "\n".join(r[0] for r in db.execute("select flds from notes"))
    db.close()
    assert "<details" in flds, "rich info must fold into <details>"
    assert "vs-src" in flds and "Charade" in flds, "Source line must always show"
    assert "Also seen in" in flds and "Wall_Street" in flds, "other occurrences listed"
    assert "Synonyms: decrease" in flds and "Related: is a change" in flds, flds[:400]
    print("anki phase2 OK -> Definition note + <details> + source line + multi-occurrence")


if __name__ == "__main__":
    test_path_a_query()
    test_loop_tool_call_cap()
    test_observe_recall_surfaces_provenance()
    test_ingest_vtt_txt_md()
    test_batch_comma_terms()
    test_anki_phase2_cards_and_extra()
    print("OK")
