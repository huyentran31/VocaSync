"""Deterministic tests for the data contract (REVIEW_CHECKLIST §A/§H)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

from schema import PersonalGraph, Node, Edge, Occurrence


def _node(occ_source="a", sentence="we reduce cost"):
    return Node(
        key="reduce#reduce.v.01", term="reduce", sense_id="reduce.v.01",
        edges=[Edge(type="synonym", target="cut", source="wordnet")],
        collocations=["cost"],
        occurrences=[Occurrence(source=occ_source, sentence=sentence, lemmas=sentence.split())],
    )


def test_upsert_merges_not_duplicates():
    g = PersonalGraph()
    g.upsert(_node("ep1", "we reduce cost"))
    g.upsert(_node("ep2", "reduce the noise"))   # same key -> merge
    assert len(g.nodes) == 1
    node = g.nodes["reduce#reduce.v.01"]
    assert len(node.occurrences) == 2            # occurrences grew
    # union edges (no dup of same (type,target))
    g.upsert(Node(key="reduce#reduce.v.01", term="reduce",
                  edges=[Edge(type="synonym", target="cut", source="wordnet"),
                         Edge(type="antonym", target="increase", source="wordnet")]))
    types = {(e.type, e.target) for e in g.nodes["reduce#reduce.v.01"].edges}
    assert ("synonym", "cut") in types and ("antonym", "increase") in types
    assert len(types) == 2                        # 'cut' not duplicated
    print("upsert merge OK:", len(node.occurrences), "occ,", len(types), "edges")


def test_recall_is_associative():
    g = PersonalGraph()
    g.upsert(_node("ep1", "we reduce cost"))
    # main node
    assert g.recall("reduce")["as_main_node"] is not None
    # edge target (peripheral)
    assert g.recall("cut")["as_related"] == ["reduce#reduce.v.01"]
    # sentence lemma
    assert g.recall("cost")["in_sentences"]
    # collocation
    assert g.recall("cost")["in_collocations"] == ["reduce#reduce.v.01"]
    assert g.has("cut") and not g.has("zzz")
    print("associative recall OK")


def test_save_load_roundtrip(tmp_path="data/_test_graph.json"):
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    g = PersonalGraph(user="tester")
    g.upsert(_node())
    g.save(tmp_path)
    g2 = PersonalGraph.load(tmp_path)
    assert g2.user == "tester" and "reduce#reduce.v.01" in g2.nodes
    os.remove(tmp_path)
    print("save/load roundtrip OK")


def test_upsert_user_edit_wins():
    g = PersonalGraph()
    base = _node()
    base.definition = "machine gloss"
    base.source_map = {"definition": "wordnet"}
    g.upsert(base)
    # re-commit with a USER-edited definition -> the human edit overrides the old value
    edited = _node()
    edited.definition = "my own wording"
    edited.source_map = {"definition": "user"}
    g.upsert(edited)
    assert g.nodes["reduce#reduce.v.01"].definition == "my own wording"
    # re-commit with a MACHINE definition while one already exists -> old value kept
    machine = _node()
    machine.definition = "another machine gloss"
    machine.source_map = {"definition": "wordnet"}
    g.upsert(machine)
    assert g.nodes["reduce#reduce.v.01"].definition == "my own wording"
    # user-edited word_type overrides too
    wt = _node()
    wt.word_type = "idiom"
    wt.source_map = {"word_type": "user"}
    g.upsert(wt)
    assert g.nodes["reduce#reduce.v.01"].word_type == "idiom"
    print("upsert user-edit precedence OK")


def test_normalize_timestamp_reuse():
    from file_utils import normalize_timestamp
    import datetime
    assert normalize_timestamp("0:5:32") == "00:05:32"
    assert normalize_timestamp(datetime.time(1, 2, 3)) == "01:02:03"
    assert normalize_timestamp("garbage") == ""
    print("normalize_timestamp OK")


def test_normalize_collocations():
    """S19 BUG-1: a `collocations` field must never explode into single characters. Guards both
    the source cause (LLM returns a STRING, not a list) and repairs an already-char-broken list."""
    from schema import normalize_collocations as nc
    # LLM returned a string -> split into phrases, NOT into characters
    assert nc("manage to make it work; try harder") == ["manage to make it work", "try harder"]
    assert nc("just one phrase") == ["just one phrase"]
    # a healthy list is preserved
    assert nc(["make it work", "go on"]) == ["make it work", "go on"]
    # a char-broken list (the bug's corrupted shape) is rejoined, not left as ['m','a','n',...]
    broke = list("manage") + [";"] + list("makeit")
    assert nc(broke) == ["manage", "makeit"]
    assert nc(None) == [] and nc("") == [] and nc(["", "  "]) == []
    print("normalize_collocations OK -> string split to phrases; char-broken list repaired")


if __name__ == "__main__":
    test_upsert_merges_not_duplicates()
    test_recall_is_associative()
    test_save_load_roundtrip()
    test_upsert_user_edit_wins()
    test_normalize_timestamp_reuse()
    test_normalize_collocations()
    print("OK")
