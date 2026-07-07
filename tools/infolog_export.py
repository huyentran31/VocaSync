"""
infolog_export.py — Export utility (read-only, NO AI, NO ffmpeg). Like obsidian_export.

A plain-text provenance log: for EACH learned word, which source(s) and WHERE in them
(timestamp) it was heard, plus the sentence. This is where the sentence text lives — the
graph stays light (it only tags source @ timestamp on hover). Mirrors the upstream
AI-Teaching tool's infolog idea.
"""

from __future__ import annotations

import os

from _common import log_tool_call   # import FIRST: puts repo root + legacy/ on sys.path


def _stamp_sec(v) -> float:
    """'HH:MM:SS[.mmm]' -> seconds; missing/unparseable -> +inf (untimed sorts last)."""
    import re
    s = str(v or "").strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)", s)
    if not m:
        return float("inf")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def export_infolog(nodes: list[dict], out_path: str) -> dict:
    """Write a provenance infolog for `nodes` (Node dicts with occurrences).

    S18 askfix (owner): entries are ordered the way you'd REWATCH the film — by (source,
    first-occurrence timestamp) instead of alphabetically — and each word shows the FULL node
    (the same fields the graph node holds: type/POS, definition, sense, topic tags,
    collocations, mnemonic, pattern, every occurrence with source @ time + sentence). Fields
    the AI authored are marked "(ai)" from the node's source_map so the provenance stays honest.
    Untimed words sort last within their source.

    Returns {"out", "nodes", "occurrences"}.
    """
    lines = [
        "VocaSync — provenance infolog",
        "Full record per word (meaning, type, tags, collocations, mnemonic) + where "
        "(source @ timestamp) it was heard, with the sentence. '(ai)' = AI-authored field.",
        "Ordered by source and timestamp (chronological — follow it while rewatching).",
        "=" * 70,
        "",
    ]

    def _node_key(n):
        occ = [o for o in (n.get("occurrences") or []) if isinstance(o, dict)]
        first_src = str(occ[0].get("source", "") if occ else "").lower()
        first_t = min((_stamp_sec(o.get("start")) for o in occ), default=float("inf"))
        return (first_src, first_t, (n.get("term") or "").lower())

    n_occ = 0
    for nd in sorted(nodes, key=_node_key):
        term = nd.get("term", "")
        smap = nd.get("source_map") or {}

        def _src(field):
            """'(ai)' / '(wordnet)' provenance tag for a field, or '' if unknown."""
            s = smap.get(field)
            return f" ({s})" if s else ""

        # Header: term + the richest available TYPE label (word_type / POS / category).
        wt = str(nd.get("word_type", "") or "word")
        pos = str(nd.get("pos", "") or "").strip()
        cat = str(nd.get("category", "") or "").strip()
        type_bits = [b for b in (wt, pos, cat) if b and b != "word"] or [wt]
        lines.append(f"{term}  [{' / '.join(dict.fromkeys(type_bits))}]")

        sense_id = str(nd.get("sense_id", "") or "").strip()
        if sense_id:
            lines.append(f"    sense: {sense_id}")
        definition = str(nd.get("definition", "") or "").strip()
        if definition:
            lines.append(f"    = {definition}{_src('definition')}")
        tags = [t for t in (nd.get("tags") or []) if str(t).strip()]
        if tags:
            lines.append(f"    tags: {', '.join(map(str, tags))}{_src('tags')}")
        collocations = [c for c in (nd.get("collocations") or []) if str(c).strip()]
        if collocations:
            lines.append(f"    collocations: {'; '.join(map(str, collocations))}{_src('collocations')}")
        pattern = str(nd.get("pattern", "") or "").strip()
        if pattern:
            lines.append(f"    pattern: {pattern}{_src('pattern')}")
        mnemonic = str(nd.get("mnemonic", "") or "").strip()
        if mnemonic:
            lines.append(f"    mnemonic: {mnemonic}{_src('mnemonic')}")
        # Related words already in the graph (edges) — the graph node's own links.
        rels = []
        for e in (nd.get("edges") or []):
            if isinstance(e, dict) and e.get("target"):
                tgt = str(e["target"]).split("#")[0]
                rels.append(f"{tgt} ({e.get('type', 'related')})" if e.get("type") else tgt)
        if rels:
            lines.append(f"    related: {', '.join(rels[:8])}")

        lines.append("    heard in:")
        occs = sorted((o for o in (nd.get("occurrences") or []) if isinstance(o, dict)),
                      key=lambda o: _stamp_sec(o.get("start")))
        for o in occs:
            src = o.get("source", "")
            stamp = o.get("start", "")
            where = f"{src} @ {stamp}" if stamp else (src or "(unknown source)")
            lines.append(f"      - {where}: {o.get('sentence', '')}")
            n_occ += 1
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    result = {"out": out_path, "nodes": len(nodes), "occurrences": n_occ}
    log_tool_call("export_infolog", {"n_nodes": len(nodes)}, result=result)
    return result
