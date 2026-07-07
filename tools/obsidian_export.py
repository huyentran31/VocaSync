"""
obsidian_export.py — utility exporter (write-local, DETERMINISTIC, no AI).

Writes the vocab Nodes as a small Obsidian vault: one Markdown note per word, with
YAML frontmatter + `[[wikilinks]]` to every edge target. Obsidian's built-in Graph
View then renders a beautiful, dark, force-directed, zoomable graph FOR FREE — no
graph-drawing code on our side. Doubles as an interoperability story (Day-2): the
same PersonalGraph speaks a second open format.

This is a utility (not an agent tool): the Mine pipeline calls it after a commit/run.
On any error it is the caller's job to ignore it — but we also never raise on a single
bad node (skip + continue), matching the project's no-crash rule.
"""

from __future__ import annotations

import os
import re

from _common import ascii_safe, log_tool_call, run_dir

# Relation -> human heading shown inside each note (order = display order).
_REL_HEADING = [
    ("synonym", "Synonyms"),
    ("antonym", "Antonyms"),
    ("is_a", "Is a"),
    ("hyponym", "Kinds of"),
    ("part_of", "Part of"),
    ("used_for", "Used for"),
    ("has_context", "Context"),
    ("collocation", "Collocations"),
]


def _coerce_nodes(units) -> list[dict]:
    if hasattr(units, "nodes") and isinstance(getattr(units, "nodes"), dict):
        return [n.model_dump() if hasattr(n, "model_dump") else n for n in units.nodes.values()]
    out = []
    for u in units or []:
        if hasattr(u, "model_dump"):
            out.append(u.model_dump())
        elif isinstance(u, dict) and "node" in u:
            out.append(u["node"])
        elif isinstance(u, dict):
            out.append(u)
    return out


def _note(nd: dict) -> str:
    """One Markdown note for a node: frontmatter + grounded links by relation."""
    term = nd.get("term", "")
    # Obsidian reads frontmatter `tags`: include the word's topic/exam tags so the user can
    # filter the Graph View by `tag:#finance` or colour a Group. ASCII-safe, spaces->_.
    ftags = ["vocabgraph"] + [
        re.sub(r"[^a-z0-9_/-]+", "_", str(t).strip().lower().replace(" ", "_")).strip("_")
        for t in (nd.get("tags") or []) if str(t).strip()
    ]
    lines = ["---",
             f"term: {term}",
             f"sense: {nd.get('sense_id') or ''}",
             f"pos: {nd.get('pos') or ''}",
             f"category: {nd.get('category') or ''}",
             f"tags: [{', '.join(dict.fromkeys(ftags))}]",
             "---",
             f"# {term}", ""]
    if nd.get("definition"):
        lines += [f"> {nd['definition']}", ""]

    edges = nd.get("edges", []) or []
    by_type: dict[str, list[str]] = {}
    for e in edges:
        et, tg = e.get("type"), (e.get("target") or "").strip()
        if not tg or et == "category":
            continue
        src = e.get("source", "")
        link = f"[[{tg}]]" + (f" <small>({src})</small>" if src == "conceptnet" else "")
        by_type.setdefault(et, []).append(link)

    for et, heading in _REL_HEADING:
        if by_type.get(et):
            lines.append(f"**{heading}:** " + ", ".join(by_type[et]))
    lines.append("")

    if nd.get("collocations"):
        lines.append("**Collocations:** " + ", ".join(nd["collocations"]))
    if nd.get("mnemonic"):
        lines.append(f"**Mnemonic:** {nd['mnemonic']}")
    occ = nd.get("occurrences") or []
    if occ:
        lines += ["", "## Seen in"]
        for o in occ[:5]:
            lines.append(f"- *{o.get('source','')}*: \"{o.get('sentence','')}\"")
    return "\n".join(lines).rstrip() + "\n"


def export_obsidian(units, run_id: str | None = None) -> str:
    """Write one .md per node into output/<run>/obsidian_vault/. Returns the vault dir."""
    nodes = _coerce_nodes(units)
    out = run_dir(run_id) if run_id else run_dir("export")
    vault = os.path.join(out, "obsidian_vault")
    os.makedirs(vault, exist_ok=True)

    written = 0
    for nd in nodes:
        try:
            term = nd.get("term") or nd.get("key")
            if not term:
                continue
            fname = ascii_safe(term) + ".md"
            with open(os.path.join(vault, fname), "w", encoding="utf-8") as f:
                f.write(_note(nd))
            written += 1
        except Exception as e:
            log_tool_call("obsidian_export", {"run_id": run_id}, error=f"skip node: {e}")
            continue

    log_tool_call("obsidian_export", {"run_id": run_id}, result={"vault": vault, "notes": written})
    return vault


if __name__ == "__main__":
    import os as _os
    import sys
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _common import load_graph
    print("vault ->", export_obsidian(load_graph(), run_id="export"))
