"""
output_utils.py — Generate output files after B5 Finalize.

Functions:
  generate_ass      — Generate .ass subtitle file (header top-left + sub bottom-center)
  generate_infolog  — Generate infolog.txt with source mapping
"""

import os
import re


def _norm_text(s: str) -> str:
    """Normalize text for matching: lowercase + collapse whitespace."""
    return " ".join(str(s).lower().split())


def _norm_match(s: str) -> str:
    """
    Aggressive normalize for highlight matching: lowercase, drop ALL punctuation
    and apostrophes, collapse whitespace. So "Couldn't!" == "couldnt" and
    "Figure something out." == "figure something out".
    """
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def _hms_to_ass_time(hms: str) -> str:
    """Convert HH:MM:SS to ASS time format H:MM:SS.cc (centiseconds)."""
    parts = hms.strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2])
    return f"{h}:{m:02d}:{s:02d}.00"


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS time H:MM:SS.cc."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s - int(s)) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _parse_srt_blocks_content(content: str) -> list[tuple[float, float, str]]:
    """Parse SRT content string (not a file path) into (start_sec, end_sec, text) tuples."""
    ts_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )
    blocks = []
    cur_start = cur_end = None
    cur_lines = []
    for line in content.splitlines():
        line = line.strip()
        m = ts_pattern.match(line)
        if m:
            if cur_start is not None and cur_lines:
                blocks.append((cur_start, cur_end, " ".join(cur_lines)))
            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
            cur_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            cur_end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            cur_lines = []
        elif line.isdigit():
            continue
        elif line:
            cur_lines.append(line)
    if cur_start is not None and cur_lines:
        blocks.append((cur_start, cur_end, " ".join(cur_lines)))
    return blocks


def _parse_srt_blocks(srt_path: str) -> list[tuple[float, float, str]]:
    """
    Parse SRT file into list of (start_sec, end_sec, text) tuples.
    Returns empty list if file not found or unreadable.
    """
    if not srt_path or not os.path.exists(srt_path):
        return []
    ts_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )
    blocks = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        current_start = current_end = None
        current_lines = []
        for line in content.splitlines():
            line = line.strip()
            m = ts_pattern.match(line)
            if m:
                if current_start is not None and current_lines:
                    blocks.append((current_start, current_end, " ".join(current_lines)))
                h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
                current_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                current_end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                current_lines = []
            elif line.isdigit():
                continue
            elif line:
                current_lines.append(line)
        if current_start is not None and current_lines:
            blocks.append((current_start, current_end, " ".join(current_lines)))
    except Exception:
        pass
    return blocks


ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Header,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,7,10,10,10,1
Style: Sub,Arial,30,&H00FFFFFF,&H000000FF,&H80000000,&H80000000,0,0,0,0,100,100,0,0,3,4,0,2,10,10,30,1
Style: Separator,Arial,20,&H0000FFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,0,0,1,2,1,8,10,10,15,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass(
    clips_info: list[dict],
    output_path: str,
    option: str = "B",
) -> None:
    """
    Generate .ass subtitle file for final video.

    clips_info: list of dicts with keys:
      - final_start_sec: float (start in final video, seconds)
      - final_end_sec: float (end in final video, seconds)
      - source_video: str (original video filename)
      - original_start: str (HH:MM:SS in source video)
      - original_end: str (HH:MM:SS in source video)
      - clip_srt_path: str (path to clip .srt file, for Option B sub text)

    option:
      "A" — Header only (source video + original timestamp), no sub text
      "B" — Header + sub text from clip SRT (default)

    ASS styles:
      Header: top-left (an7), fs14 — source video + original timestamp
      Sub:    bottom-center (an2), fs20 — dialogue from clip SRT
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dialogue_lines = []

    total = len(clips_info)
    for i, clip in enumerate(clips_info):
        start = _seconds_to_ass_time(clip["final_start_sec"])
        end = _seconds_to_ass_time(clip["final_end_sec"])
        source = clip.get("source_video", "")
        orig_start = clip.get("original_start", "")
        orig_end = clip.get("original_end", "")

        header_text = f"{source} | {orig_start}→{orig_end}"
        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Header,,0,0,0,,{header_text}"
        )

        if i > 0:
            sep_end_sec = clip["final_end_sec"]
            sep_end = _seconds_to_ass_time(sep_end_sec)
            dialogue_lines.append(
                f"Dialogue: 0,{start},{sep_end},Separator,,0,0,0,,► Clip {i + 1} / {total}"
            )

        if option == "B":
            blocks = _parse_srt_blocks(clip.get("clip_srt_path", ""))
            # Highlight the sub line(s) containing the TARGET PHRASE (language_item) —
            # short and robust to match. Fall back to example_sub if no language_item.
            # Punctuation/case ignored. ';' separates merged multi-item clips.
            raw_focus = str(clip.get("language_item") or clip.get("example_sub") or "")
            focus_items = [_norm_match(p) for p in raw_focus.split(";")]
            focus_items = [f for f in focus_items if f]
            clip_dur = clip["final_end_sec"] - clip["final_start_sec"]
            for blk_start, blk_end, text in blocks:
                # Clamp subtitle timing inside the clip so it never bleeds onto
                # the following "Next" clip or the next segment.
                if blk_start >= clip_dur:
                    continue
                blk_end = min(blk_end, clip_dur)
                bs = _seconds_to_ass_time(clip["final_start_sec"] + blk_start)
                be = _seconds_to_ass_time(clip["final_start_sec"] + blk_end)
                blk_norm = _norm_match(text)
                if blk_norm and any(f in blk_norm for f in focus_items):
                    # Whole focus line: bold + bigger + yellow (inline override only
                    # affects this Dialogue line; next line resets to Sub style).
                    text = r"{\b1\fs40\c&H00FFFF&}" + text
                dialogue_lines.append(
                    f"Dialogue: 0,{bs},{be},Sub,,0,0,0,,{text}"
                )

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(dialogue_lines))
        f.write("\n")


def generate_infolog(
    clips_info: list[dict],
    output_path: str,
    group_col: str = "none",
) -> None:
    """
    Generate infolog.txt with source mapping for each segment in final video.

    clips_info: list of dicts with keys:
      - final_start: str HH:MM:SS (in final video after reset)
      - final_end: str HH:MM:SS
      - source_video: str
      - original_start: str HH:MM:SS
      - original_end: str HH:MM:SS
      - clip: str (clip filename)
      - group_value: str (value of group_col for this clip)
      - original_word: str
      - language_item: str
      - language_tag: str
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "AI Teaching System — Flow 1 | Final Video Index",
        f"Generated: {_now_str()}",
        f"Group by: {group_col}",
        "=" * 60,
        "",
    ]

    for idx, clip in enumerate(clips_info, start=1):
        group_val = clip.get("group_value", "")
        orig_word = clip.get("original_word", "")
        lang_item = clip.get("language_item", "")
        lang_tag = clip.get("language_tag", "")

        word_line = ""
        if orig_word or lang_item or lang_tag:
            word_line = f"  Word:     {orig_word} | language_item: {lang_item} | tag: {lang_tag}"

        ai_note = clip.get("ai_note", "").strip()
        example_sub = clip.get("example_sub", "").strip()

        block = [
            f"[{idx}]",
            f"  Output:   {clip.get('final_start', '')} → {clip.get('final_end', '')}",
            f"  Source:   {clip.get('source_video', '')}",
            f"  Original: {clip.get('original_start', '')} → {clip.get('original_end', '')}",
            f"  Clip:     {clip.get('clip', '')}",
        ]
        if group_val:
            block.append(f"  Group:    {group_val}")
        if word_line:
            block.append(word_line)
        if ai_note:
            block.append(f"  AI note:  {ai_note}")
        if example_sub:
            # Check per language_item against its individual (pre-merge) SRT when available.
            # Falls back to merged SRT so single-item clips and regen still work.
            lang_items_list = [p.strip() for p in lang_item.split(";") if p.strip()] or ([lang_item] if lang_item else [])
            per_item_contents = clip.get("_per_item_srt_contents", [])
            merged_srt_blocks = None  # lazy-load once if needed as fallback

            not_found = []
            multi = []  # items found more than once
            for idx_i, item in enumerate(lang_items_list):
                focus = _norm_match(item)
                if not focus:
                    continue
                if idx_i < len(per_item_contents) and per_item_contents[idx_i]:
                    item_blocks = _parse_srt_blocks_content(per_item_contents[idx_i])
                else:
                    if merged_srt_blocks is None:
                        merged_srt_blocks = _parse_srt_blocks(clip.get("clip_srt_path", ""))
                    item_blocks = merged_srt_blocks
                hits = [text for _, _, text in item_blocks if focus in _norm_match(text)]
                if not hits:
                    not_found.append(item)
                elif len(hits) > 1:
                    multi.append((item, hits))

            if multi or not_found:
                notes = []
                if multi:
                    for item, hits in multi:
                        snippets = " / ".join(f'"{h}"' for h in hits)
                        notes.append(f'"{item}" found {len(hits)}x — compare with students: {snippets}')
                if not_found:
                    not_found_str = "; ".join(f'"{x}"' for x in not_found)
                    notes.append(f"[warning] not found in SRT — highlight skipped: {not_found_str}")
                block.append(f"  Ex note:  {' | '.join(notes)}")
            block.append(f"  Ex sub:   {example_sub}")
        block.append("")

        lines.extend(block)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
