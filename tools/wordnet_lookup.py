"""
wordnet_lookup.py — Tool #4 (read, cheap, DETERMINISTIC).

Return ALL WordNet senses of a term, each with its grounded relations. This is the
anti-hallucination backbone (Day-4): synonyms / antonyms / is_a / hyponym / category
come straight from WordNet — the LLM is NOT allowed to invent them. `enrich` later
PICKS one of these senses and only then adds uncertain, AI-flagged fields.

Every edge is emitted as a schema.Edge with source="wordnet" so it can be dropped
into a Node untouched. Edge.type stays within EDGE_TYPES (schema.py).

On error / unknown word: return found=False with senses=[] (docs/TOOLS.md).
"""

from __future__ import annotations

from _common import log_tool_call
from schema import Edge, EDGE_TYPES

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "term": {"type": "string", "description": "Word/lemma to look up in WordNet."},
        "max_per_relation": {"type": "integer", "description": "Cap items per relation (default 6)."},
    },
    "required": ["term"],
}

# WordNet pos code -> readable pos (aligns with spaCy-ish tags used elsewhere)
_POS = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}


def _lemma_surface(synset) -> str:
    """The synset's display lemma, e.g. 'reduce.v.01' -> 'reduce'."""
    return synset.name().split(".")[0].replace("_", " ")


def wordnet_lookup(term: str, max_per_relation: int = 6) -> dict:
    """Return {"term","found","senses":[...]}.

    Each sense:
      {sense_id, pos, category, definition, examples,
       synonyms, antonyms, hypernyms, hyponyms,
       edges: [Edge-as-dict ...]}   # ready to merge into a Node
    """
    args = {"term": term, "max_per_relation": max_per_relation}

    # BATCH input (S17-5.1a): a comma-separated string means MULTIPLE terms (no English
    # lemma contains a comma) — the old behavior looked up the whole string and silently
    # returned found=False/0 senses. Split and look up each term instead; the single-term
    # return shape is unchanged, batch adds {"batch": {term: result}, "note": ...}.
    terms = [t.strip() for t in str(term or "").split(",") if t.strip()]
    if len(terms) > 1:
        out = {"term": term, "found": False, "senses": [], "batch": {},
               "note": (f"input contained {len(terms)} comma-separated terms — wordnet_lookup "
                        "takes ONE term; each was looked up separately (see 'batch')")}
        for t in terms:
            res = wordnet_lookup(t, max_per_relation)
            out["batch"][t] = res
            out["found"] = out["found"] or bool(res.get("found"))
        log_tool_call("wordnet_lookup", {"term": term, "batch": len(terms)},
                      result={"found": out["found"], "terms": terms})
        return out

    try:
        from nltk.corpus import wordnet as wn
    except Exception as e:  # nltk/data missing is a setup issue, but stay non-fatal here
        log_tool_call("wordnet_lookup", args, error=f"wordnet unavailable: {e}")
        return {"term": term, "found": False, "senses": []}

    try:
        query = term.strip().lower().replace(" ", "_")
        synsets = wn.synsets(query)
        senses = []
        for ss in synsets:
            head = _lemma_surface(ss)

            synonyms, antonyms, edges = [], [], []

            # synonyms = other lemmas in the same synset
            for lem in ss.lemmas():
                name = lem.name().replace("_", " ")
                if name.lower() != head.lower() and name not in synonyms:
                    synonyms.append(name)
                # antonyms hang off lemmas
                for ant in lem.antonyms():
                    a = ant.name().replace("_", " ")
                    if a not in antonyms:
                        antonyms.append(a)

            hypernyms = [_lemma_surface(h) for h in ss.hypernyms()]
            hyponyms = [_lemma_surface(h) for h in ss.hyponyms()]
            # holonyms = the WHOLE this sense is A PART OF (tusk -> elephant).
            # gather all three flavours: part / member / substance holonyms.
            holonyms, _seen_hol = [], set()
            for h in (ss.part_holonyms() + ss.member_holonyms() + ss.substance_holonyms()):
                name = _lemma_surface(h)
                if name not in _seen_hol:
                    _seen_hol.add(name)
                    holonyms.append(name)
            category = ss.lexname()  # e.g. "verb.change" — used for clustering

            # --- build deterministic edges (source="wordnet") ---
            for s in synonyms[:max_per_relation]:
                edges.append(Edge(type="synonym", target=s, source="wordnet"))
            for a in antonyms[:max_per_relation]:
                edges.append(Edge(type="antonym", target=a, source="wordnet"))
            for h in hypernyms[:max_per_relation]:
                edges.append(Edge(type="is_a", target=h, source="wordnet"))
            for h in hyponyms[:max_per_relation]:
                edges.append(Edge(type="hyponym", target=h, source="wordnet"))
            for h in holonyms[:max_per_relation]:   # "A PART OF" the whole
                edges.append(Edge(type="part_of", target=h, source="wordnet"))
            if category:
                edges.append(Edge(type="category", target=category, source="wordnet"))

            assert all(e.type in EDGE_TYPES for e in edges)  # locked vocabulary

            senses.append({
                "sense_id": ss.name(),
                "pos": _POS.get(ss.pos(), ss.pos()),
                "category": category,
                "definition": ss.definition(),
                "examples": ss.examples()[:3],
                "synonyms": synonyms[:max_per_relation],
                "antonyms": antonyms[:max_per_relation],
                "hypernyms": hypernyms[:max_per_relation],
                "hyponyms": hyponyms[:max_per_relation],
                "holonyms": holonyms[:max_per_relation],
                "edges": [e.model_dump() for e in edges],
            })

        out = {"term": term, "found": bool(senses), "senses": senses}
        log_tool_call("wordnet_lookup", args, result={"senses": len(senses)})
        return out
    except Exception as e:
        log_tool_call("wordnet_lookup", args, error=str(e))
        return {"term": term, "found": False, "senses": []}


if __name__ == "__main__":
    import json
    import sys

    t = sys.argv[1] if len(sys.argv) > 1 else "gas"
    r = wordnet_lookup(t)
    print(f"{t}: found={r['found']} senses={len(r['senses'])}")
    for s in r["senses"][:4]:
        print(f"  {s['sense_id']:18} [{s['category']}] {s['definition'][:60]}")
