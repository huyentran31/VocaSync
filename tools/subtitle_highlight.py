"""
subtitle_highlight.py — Export utility (read-only, NO AI, NO ffmpeg). Like obsidian_export.

Take a source .srt + the learner's mined target terms and emit an .ass subtitle where
every line that CONTAINS a target term is highlighted in yellow (whole-line, the same
behaviour as the upstream AI-Teaching tool's generate_ass). Load it over the video in
VLC/MPC ("watch the episode, your learned phrases glow") — a "learn-through-play" view.

Matching reuses the EXACT same normalization the words already passed through in
extract_vocab._grounded ([a-z0-9]+ lowercase, substring), so any term confirmed to be
grounded in the transcript is guaranteed findable here — no separate matching rules.

Generating the .ass needs only the .srt (no MP4); the MP4 is needed only to WATCH it.
"""

from __future__ import annotations

import os

from _common import log_tool_call   # import FIRST: puts repo root + legacy/ on sys.path
from output_utils import _parse_srt_blocks, _norm_match, _seconds_to_ass_time, ASS_HEADER

# bold + bigger + yellow — the same inline override generate_ass uses for the target line.
_HL = r"{\b1\fs40\c&H00FFFF&}"


def export_highlighted_ass(srt_path: str, terms: list[str], out_path: str) -> dict:
    """Write `out_path` (.ass): every transcript line containing a target term is yellow.

    terms: surface/lemma strings of the mined vocab (surface preferred — see module doc).
    Returns {"out", "lines", "highlighted"}.
    """
    # normalize once; longest first is irrelevant for whole-line but keeps the set clean
    norm_terms = sorted({n for t in terms if (n := _norm_match(t))}, key=len, reverse=True)
    blocks = _parse_srt_blocks(srt_path)

    lines, n_hl = [], 0
    for start, end, text in blocks:
        bs, be = _seconds_to_ass_time(start), _seconds_to_ass_time(end)
        blk = _norm_match(text)
        if blk and any(nt in blk for nt in norm_terms):
            text = _HL + text          # whole-line highlight; next line resets to Sub style
            n_hl += 1
        lines.append(f"Dialogue: 0,{bs},{be},Sub,,0,0,0,,{text}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(lines))
        f.write("\n")

    result = {"out": out_path, "lines": len(blocks), "highlighted": n_hl}
    log_tool_call("export_highlighted_ass",
                  {"srt": srt_path, "n_terms": len(norm_terms)}, result=result)
    return result


if __name__ == "__main__":
    import json
    import sys

    srt = sys.argv[1]
    # terms from a pending_drafts.json (surface field) or a comma list
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    out = sys.argv[3] if len(sys.argv) > 3 else "highlighted.ass"
    if arg.endswith(".json"):
        pend = json.load(open(arg, encoding="utf-8"))
        terms = [v.get("surface") or v["node"]["term"] for v in pend.values()]
    else:
        terms = [t for t in arg.split(",") if t.strip()]
    print(export_highlighted_ass(srt, terms, out))
