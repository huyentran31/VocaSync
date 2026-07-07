"""
schema.py — The data contract for the whole project (the "spec is gold" keystone).

Design principles baked in here:
  • Personal graph that GROWS over time  → identity by stable `key`; every new
    encounter appends an `Occurrence` instead of duplicating a node.
  • Deterministic-first, AI-flagged       → `source_map` records where each field
    came from ("wordnet" | "ai" | "spacy" | "user"); AI fields are reviewable.
  • Associative recall (not exact-match)  → recall() scans main nodes, edge targets,
    occurrence lemmas, and collocations — surfaces a word even as a *peripheral* node.
  • Dependency-light                       → only pydantic; lemmatization/IO happen
    in tools, lemmas are stored so recall stays a cheap membership check.

Everything downstream (tools, agent, graph, anki) reads/writes these models.
"""

from __future__ import annotations
import re
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Collocations normaliser (S19 BUG-1) — shared by enrich / review_io / app
# --------------------------------------------------------------------------- #

def normalize_collocations(value) -> list[str]:
    """Coerce a `collocations` field into a clean list[str], guarding two corruptions:

      1. LLM returned a STRING ("manage to make it work; try harder") instead of a list.
         Iterating that string char-by-char (the old `for c in <str>`) exploded it into
         ['m','a','n',...] — the S19 "collocations broke into single characters" bug.
         Fix at source: split the string on ';'/',' into phrases.
      2. A already-char-broken LIST (['m','a','n',...]) reaching a display/commit path:
         detect a run of single-character items and rejoin them, then re-split on any
         surviving ';'/',' separators. Best-effort — the spaces were stripped when the
         corruption happened, so recovered phrases lack internal spaces (old data).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
    if isinstance(value, (list, tuple)):
        items = [str(c) for c in value]
        # char-broken: 3+ items, every item is 0/1 char -> was a string iterated per-char
        if len(items) >= 3 and all(len(c.strip()) <= 1 for c in items):
            rejoined = "".join(items)
            return [p.strip() for p in re.split(r"[;,]", rejoined) if p.strip()]
        return [c.strip() for c in items if isinstance(c, str) and c.strip()]
    return []


# --------------------------------------------------------------------------- #
# Leaf models
# --------------------------------------------------------------------------- #

class Media(BaseModel):
    """A piece of media attached to an occurrence. type='none' => text-only card."""
    type: str = "none"          # "video" | "audio" | "image" | "none"
    path: str = ""              # ASCII-safe path/filename (Anki normalizes names)


class Occurrence(BaseModel):
    """ONE time the term was met. The personal graph grows by appending these."""
    source: str                 # "SpiritedAway_ep1" | "podcast_x" | "manual_import"
    sentence: str               # the line the term appeared in
    surface: str = ""           # the ORIGINAL word form as it appeared here ("figured out",
                                #   "emissions") — the node's `term` is the lemma, but every
                                #   sighting keeps its real form so no source data is lost.
    lemmas: list[str] = []      # precomputed lemmas of `sentence` -> cheap recall
    media: Media = Field(default_factory=Media)
    added_at: str = ""          # ISO date (passed in by caller, never auto-generated)
    start: str = ""             # "HH:MM:SS" in the SOURCE media where this line occurs (provenance)
    end: str = ""               # "HH:MM:SS" end of that line (empty when timestamps are unknown)
    video_ref: str = ""         # link to source/teaching video, if any
    card_id: str = ""           # GUID of the Anki card generated -> reopen old card


class Edge(BaseModel):
    """A relation to another node. `target` is another node's `key` (or a lemma)."""
    type: str                   # synonym|antonym|is_a|hyponym|category|collocation
    target: str
    source: str                 # "wordnet" | "ai" | "spacy"


class Node(BaseModel):
    """One vocabulary entry in the personal graph."""
    key: str                    # STABLE id = f"{lemma}#{sense_id}" e.g. "reduce#reduce.v.01"
    term: str                   # surface/lemma shown to the learner
    word_type: str = "word"     # word|phrasal_verb|idiom|collocation|slang (from extract_vocab)
    sense_id: str | None = None # WordNet synset name after disambiguation
    pos: str | None = None      # part of speech (spaCy)            [deterministic]
    category: str | None = None # WordNet lexname -> used for clustering [deterministic]
    ipa: str | None = None      # pronunciation                     [deterministic]
    definition: str | None = None  # WordNet gloss of the chosen sense [deterministic]
    edges: list[Edge] = []      # deterministic relations (mostly WordNet)
    # --- AI-enriched (flagged via source_map; need human review) ---
    collocations: list[str] = []
    mnemonic: str | None = None
    pattern: str | None = None
    # Topic / exam tags (e.g. ["finance", "IELTS"]). AI proposes from the focus topic,
    # the learner edits in the in-app review table. Ordered by relevance (strongest first). Drives the
    # graph's group/filter ("show only finance"). Origin tracked in source_map['tags'].
    tags: list[str] = []
    # --- growth + state ---
    occurrences: list[Occurrence] = []
    status: str = "new"         # new|learning|known (reserved for i+1 — Vision)
    source_map: dict[str, str] = {}  # field name -> origin ("wordnet"|"ai"|"spacy"|"user")


# --------------------------------------------------------------------------- #
# The persistent store (one JSON file per user; loaded + merged each session)
# --------------------------------------------------------------------------- #

class PersonalGraph(BaseModel):
    user: str = "default"
    version: int = 1
    nodes: dict[str, Node] = {}      # key -> Node (dict => O(1) merge/dedup)

    # ---- growth ----------------------------------------------------------- #
    def upsert(self, node: Node) -> Node:
        """Merge a node into the graph. Same key => combine, don't duplicate.

        Merge policy: append new occurrences; union edges/collocations; fill
        empty deterministic fields; keep existing AI fields unless empty.
        """
        existing = self.nodes.get(node.key)
        if existing is None:
            self.nodes[node.key] = node
            return node
        # combine occurrences (the "grows over time" part), deduped by
        # (source, start, sentence, surface) so re-committing the SAME media doesn't stack
        # identical sightings (the word would otherwise show the same "source @ time" twice).
        seen_occ = {(o.source, o.start, o.sentence, o.surface) for o in existing.occurrences}
        for o in node.occurrences:
            ident = (o.source, o.start, o.sentence, o.surface)
            if ident not in seen_occ:
                seen_occ.add(ident)
                existing.occurrences.append(o)
        # union edges (dedup by (type, target))
        seen = {(e.type, e.target) for e in existing.edges}
        existing.edges.extend(e for e in node.edges if (e.type, e.target) not in seen)
        # union collocations
        for c in node.collocations:
            if c not in existing.collocations:
                existing.collocations.append(c)
        # union tags (keep order; first-seen wins so the strongest tag stays at front)
        for t in node.tags:
            if t not in existing.tags:
                existing.tags.append(t)
        # fill empty deterministic / AI fields; a USER-edited incoming value always
        # wins over a machine value (human edits must survive a re-commit of the same key)
        for f in ("sense_id", "pos", "category", "ipa", "mnemonic", "pattern",
                  "definition", "word_type"):
            if node.source_map.get(f) == "user" and getattr(node, f):
                setattr(existing, f, getattr(node, f))
            elif not getattr(existing, f) and getattr(node, f):
                setattr(existing, f, getattr(node, f))
        existing.source_map.update(node.source_map)
        return existing

    # ---- associative recall (NOT exact match) ----------------------------- #
    def recall(self, lemma: str) -> dict:
        """Find every trace of `lemma` in the graph so the learner can connect it.

        Surfaces it as: a main node, a target inside other nodes' edges (peripheral),
        inside occurrence sentences (via stored lemmas), and inside collocations.
        Returns a dict of hits (empty lists if none). Cheap: pure scan, no NLP here.
        """
        lemma = lemma.strip().lower()
        hits = {"as_main_node": None, "as_related": [], "in_sentences": [], "in_collocations": []}
        for key, node in self.nodes.items():
            if node.term.strip().lower() == lemma:
                hits["as_main_node"] = node
            if any(e.target.strip().lower() == lemma for e in node.edges):
                hits["as_related"].append(key)
            for occ in node.occurrences:
                if lemma in (l.lower() for l in occ.lemmas):
                    hits["in_sentences"].append({
                        "node": key, "source": occ.source, "sentence": occ.sentence,
                        "start": occ.start, "end": occ.end,   # NEW: where in the source media
                    })
            if any(lemma == c.strip().lower() for c in node.collocations):
                hits["in_collocations"].append(key)
        return hits

    def has(self, lemma: str) -> bool:
        h = self.recall(lemma)
        return bool(h["as_main_node"] or h["as_related"] or h["in_sentences"] or h["in_collocations"])

    # ---- persistence (load at session start, save after) ------------------ #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "PersonalGraph":
        import os
        if not os.path.exists(path):
            return cls()                      # fresh graph on first run
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


# Edge type vocabulary (locked) — keep tools consistent.
#   WordNet core : synonym · antonym · is_a · hyponym · category · part_of
#   AI/enrich    : collocation
#   ConceptNet   : used_for (functional) · has_context (life domain) — kept SEPARATE from
#                  `category` (which is the WordNet lexname used for Louvain clustering).
EDGE_TYPES = ("synonym", "antonym", "is_a", "hyponym", "category", "collocation",
              "part_of", "used_for", "has_context")
