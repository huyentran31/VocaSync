"""
ingest_transcript.py — Tool #2 (read, medium cost). Deterministic pipeline step.

Turn a media file into timestamped transcript SEGMENTS that extract_vocab can mine
and make_anki can cut clips from.

Two inputs, one output contract:
  • .srt given        → parse it directly (no transcription needed) — cheap, testable.
  • video/audio given → transcribe with Whisper (REUSES legacy/whisper_utils), then
                        parse the produced SRT. The SRT is written under output/<run>/.

Whisper + ffmpeg are SYSTEM dependencies (AGENTS.md §4): a missing file or a missing
binary is a system-error → HALT with a clear message (do NOT silently return empty).

Output segment: {"idx", "start", "end", "start_sec", "end_sec", "text"} where
start/end are "HH:MM:SS" (what ffmpeg.cut_clip expects).
"""

from __future__ import annotations

import os
import re

from _common import (SystemError_, log_tool_call, new_run_id, run_dir)

# legacy reuse (on sys.path via _common)
from whisper_utils import transcribe_to_srt  # noqa: E402

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Path to a video/audio file, or a transcript "
                   "(.srt / .vtt with timings, or .txt / .md as plain text without timings)."},
        "language": {"type": "string", "description": "Optional ISO language hint for Whisper (e.g. 'en')."},
        "run_id": {"type": "string", "description": "Optional run id; a new one is created if omitted."},
    },
    "required": ["source"],
}

_SRT_EXT = {".srt"}
_VTT_EXT = {".vtt"}
_TEXT_EXT = {".txt", ".md"}
_MEDIA_EXT = {".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def _sec_to_hms(sec: float) -> str:
    sec = int(max(0.0, sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def _vtt_ts_to_sec(ts: str) -> float:
    """WebVTT timestamp 'HH:MM:SS.mmm' or 'MM:SS.mmm' -> seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    else:
        return 0.0
    return h * 3600 + m * 60 + s


_VTT_TIME = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})")


def _parse_vtt(path: str) -> list[dict]:
    """Parse a WebVTT file into timestamped segments (same shape as _parse_srt). Cue ids,
    the WEBVTT header, NOTE blocks and cue-setting suffixes are ignored. ~Python-only."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.read()
    segs, idx = [], 1
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines or lines[0].upper().startswith("WEBVTT") or lines[0].upper().startswith("NOTE"):
            continue
        m = None
        text_lines = []
        for ln in lines:
            hit = _VTT_TIME.search(ln)
            if hit and m is None:
                m = hit
            elif hit is None and m is not None:
                text_lines.append(ln)
        if not m:
            continue
        text = " ".join(text_lines).strip()
        if not text:
            continue
        s, e = _vtt_ts_to_sec(m.group(1)), _vtt_ts_to_sec(m.group(2))
        segs.append({"idx": idx, "start_sec": s, "end_sec": e,
                     "start": _sec_to_hms(s), "end": _sec_to_hms(e), "text": text})
        idx += 1
    return segs


def _strip_md(line: str) -> str:
    """Reduce a Markdown line to plain text (headings, emphasis, links, inline code, bullets)."""
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)      # [text](url) -> text
    line = re.sub(r"[*_`~]+", "", line)                        # emphasis / code marks
    line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)              # heading hashes
    line = re.sub(r"^\s*[-*+]\s+", "", line)                   # bullet markers
    line = re.sub(r"^\s*>\s?", "", line)                       # blockquote
    return line.strip()


def _parse_plaintext(path: str, markdown: bool) -> list[dict]:
    """Read a .txt/.md file as a raw transcript: each non-empty line becomes a segment with
    NO timestamp (start/end empty — downstream tolerates this). Markdown syntax is stripped."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    segs, idx = [], 1
    for ln in lines:
        text = _strip_md(ln) if markdown else ln.strip()
        if not text:
            continue
        segs.append({"idx": idx, "start_sec": "", "end_sec": "",
                     "start": "", "end": "", "text": text})
        idx += 1
    return segs


def _parse_srt(path: str) -> list[dict]:
    """Parse an SRT file into segments. Reuses output_utils' robust block parser."""
    from output_utils import _parse_srt_blocks  # (start_sec, end_sec, text) tuples
    segs = []
    for i, (s, e, text) in enumerate(_parse_srt_blocks(path), start=1):
        text = (text or "").strip()
        if not text:
            continue
        segs.append({
            "idx": i, "start_sec": s, "end_sec": e,
            "start": _sec_to_hms(s), "end": _sec_to_hms(e), "text": text,
        })
    return segs


def ingest_transcript(source: str, language: str | None = None, run_id: str | None = None) -> dict:
    """Transcribe/parse `source` into segments.

    Returns {"run_id", "srt_path", "segments": [...], "full_text", "source"}.
    Raises SystemError_ on missing file, unsupported type, or missing Whisper/ffmpeg.
    """
    args = {"source": source, "language": language, "run_id": run_id}
    try:
        if not source or not os.path.exists(source):
            raise SystemError_(f"Source not found: {source!r}")

        run_id = run_id or new_run_id()
        out = run_dir(run_id)
        ext = os.path.splitext(source)[1].lower()

        if ext in _SRT_EXT:
            srt_path = source
            segments = _parse_srt(srt_path)
        elif ext in _VTT_EXT:
            srt_path = source            # keep the source path; .ass export tolerates non-.srt
            segments = _parse_vtt(source)
        elif ext in _TEXT_EXT:
            srt_path = source
            segments = _parse_plaintext(source, markdown=(ext == ".md"))
        elif ext in _MEDIA_EXT:
            srt_path = os.path.join(out, "transcript.srt")
            ok = transcribe_to_srt(source, srt_path, language=language)
            if not ok or not os.path.exists(srt_path):
                # whisper_utils swallows ImportError/runtime errors → False.
                raise SystemError_(
                    "Whisper transcription failed. Ensure `openai-whisper` is installed "
                    "and `ffmpeg` is on PATH (set FFMPEG_PATH in .env). "
                    "Alternatively pass an existing .srt to ingest_transcript."
                )
            segments = _parse_srt(srt_path)
        else:
            raise SystemError_(
                f"Unsupported source type {ext!r}. Give a video/audio file, or a transcript "
                f"(.srt / .vtt / .txt / .md)."
            )

        result = {
            "run_id": run_id,
            "srt_path": srt_path,
            "segments": segments,
            "full_text": " ".join(s["text"] for s in segments),
            "source": source,
        }
        # S17 ①: stash the transcript so stage_for_review can verify sentence∈transcript
        # on the agent write path (keyed by source basename). No-crash; degrades to no cache.
        try:
            from _common import cache_transcript
            # S18 #2: also stash srt_path so commit can recover clip timings for agent-staged
            # words when the sidebar last_srt is empty (mine-via-chat). Only timestamped
            # segment files help clip-cutting; a plain .txt/.md source has no timings anyway.
            cache_transcript(source, result["full_text"], srt_path=result.get("srt_path", ""))
        except Exception:
            pass
        log_tool_call("ingest_transcript",
                      {"source": source, "run_id": run_id},
                      result={"segments": len(segments), "srt_path": srt_path})
        return result
    except SystemError_ as e:
        log_tool_call("ingest_transcript", args, error=str(e))
        raise
    except Exception as e:
        # unexpected → treat as system-error (don't pretend success)
        log_tool_call("ingest_transcript", args, error=str(e))
        raise SystemError_(f"ingest_transcript failed: {e}")


if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(ingest_transcript(src), ensure_ascii=False, indent=2)[:1500])
