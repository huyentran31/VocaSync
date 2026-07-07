"""
Deterministic test for the new WordNet "part_of" relation (meronym/holonym).

A tusk is A PART OF an elephant. WordNet stores this as a holonym of tusk.n.02
(part_holonyms -> elephant.n.01). wordnet_lookup must surface it as a
schema.Edge(type="part_of", target="elephant", source="wordnet") — deterministic,
no AI. Direction matters: tusk -> elephant (the whole), never the reverse.

Offline, no key. Run: python tests/test_part_of.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools")):
    sys.path.insert(0, p)

from wordnet_lookup import wordnet_lookup
from schema import EDGE_TYPES


def test_tusk_is_part_of_elephant():
    assert "part_of" in EDGE_TYPES                      # vocabulary unlocked

    r = wordnet_lookup("tusk")
    assert r["found"] is True

    part_of = [e for s in r["senses"] for e in s["edges"] if e["type"] == "part_of"]
    assert part_of, "expected at least one part_of edge for 'tusk'"

    # every part_of edge is deterministic (from WordNet, not AI)
    assert all(e["source"] == "wordnet" for e in part_of)

    targets = {e["target"].lower() for e in part_of}
    assert any("elephant" in t for t in targets), f"elephant not among holonyms: {targets}"

    print(f"part_of OK -> tusk part_of {sorted(targets)}")


if __name__ == "__main__":
    test_tusk_is_part_of_elephant()
    print("OK")
