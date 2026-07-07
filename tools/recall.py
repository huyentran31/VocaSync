"""
recall.py — Tool #1 (read, cheap).

Find every trace of a word in the learner's PersonalGraph: as a main node, as a
target inside other nodes' edges (peripheral), inside occurrence sentences (via
stored lemmas), and inside collocations. ASSOCIATIVE, not exact-match.

This is the FIRST tool every skill calls (AGENTS.md §5 "recall first"): reuse what
the learner already knows before fetching from WordNet/AI again.

The actual scan lives in schema.PersonalGraph.recall — this tool is a thin,
crash-proof wrapper that loads the graph, logs the call, and serializes hits.

On error: return empty hits (docs/TOOLS.md) — recall must never crash a turn.
"""

from __future__ import annotations

from _common import GRAPH_PATH, load_graph, log_tool_call

# ---- JSON schema (Day-2: each tool declares its in/out contract) ------------ #
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "lemma": {"type": "string", "description": "Word/lemma to look up (case-insensitive)."},
    },
    "required": ["lemma"],
}

EMPTY_HITS = {"as_main_node": None, "as_related": [], "in_sentences": [], "in_collocations": []}


def _split_terms(text: str) -> list[str]:
    """Comma-separated input -> clean per-term list ('a, b,,c ' -> ['a','b','c'])."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def recall(lemma: str, graph_path: str = GRAPH_PATH) -> dict:
    """Return associative hits for `lemma` from the PersonalGraph.

    Returns a dict:
      {
        "found": bool,
        "as_main_node": <Node dict or None>,
        "as_related": [<node key>, ...],          # lemma appears as an edge target
        "in_sentences": [{"node","source","sentence"}, ...],
        "in_collocations": [<node key>, ...],
      }
    Cheap, read-only. Any failure degrades to empty hits (never raises).

    BATCH input (S17-5.1a): the agent sometimes packs many terms into one call
    ("make it work, go on, …"), which used to silently return found=False for the
    whole string. A comma means MULTIPLE terms (no stored term contains a comma) →
    split and recall each, returning {"found": <any>, "batch": {term: <hits>},
    "note": ...}. The single-term shape above is unchanged.
    """
    terms = _split_terms(lemma)
    if len(terms) > 1:
        out = {"found": False, "batch": {}, **EMPTY_HITS,
               "note": (f"input contained {len(terms)} comma-separated terms — recall takes "
                        "ONE lemma; each term was recalled separately (see 'batch')")}
        for t in terms:
            hit = recall(t, graph_path)
            out["batch"][t] = hit
            out["found"] = out["found"] or bool(hit.get("found"))
        log_tool_call("recall", {"lemma": lemma, "batch": len(terms)},
                      result={"found": out["found"], "terms": terms})
        return out
    try:
        graph = load_graph(graph_path)
        hits = graph.recall(lemma)
        # serialize the Node (schema model) so the result is plain JSON
        main = hits.get("as_main_node")
        out = {
            "found": graph.has(lemma),
            "as_main_node": main.model_dump() if main is not None else None,
            "as_related": hits.get("as_related", []),
            "in_sentences": hits.get("in_sentences", []),
            "in_collocations": hits.get("in_collocations", []),
        }
        # S18 (owner request): the REVIEW QUEUE is part of the learner's memory too — a word
        # can be STAGED (awaiting approval) but not yet committed to the graph. Surface it
        # (additive key) so the agent never tells the learner "you haven't learned this"
        # while the word sits in the queue. No-crash; absent queue -> key omitted.
        try:
            import review_io
            low = str(lemma or "").strip().lower()
            q = []
            for k, v in review_io.load_pending().items():
                if k == "_meta" or not isinstance(v, dict):
                    continue
                node = v.get("node", {}) if isinstance(v.get("node"), dict) else {}
                if str(node.get("term", "")).strip().lower() == low:
                    q.append({"key": k,
                              "flagged_ungrounded": "ungrounded" in (v.get("ai_fields") or [])})
            if q:
                out["in_review_queue"] = q
        except Exception:
            pass
        log_tool_call("recall", {"lemma": lemma}, result=out)
        return out
    except Exception as e:
        log_tool_call("recall", {"lemma": lemma}, error=str(e))
        return {"found": False, **EMPTY_HITS}


if __name__ == "__main__":
    import json
    import sys

    word = sys.argv[1] if len(sys.argv) > 1 else "reduce"
    print(json.dumps(recall(word), ensure_ascii=False, indent=2))
