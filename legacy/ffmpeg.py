"""
ffmpeg.py — Video processing via ffmpeg.

Functions:
  cut_clip           — Cut a clip from source video by timestamp
  merge_video        — Merge list of clips into final video
  reset_timestamp    — Reset final video timestamp to 00:00:00
  delete_unused_clips — Delete non-approved clips and SRT files
  get_clip_duration  — Get clip duration in seconds via ffprobe
"""

import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def _run_ffmpeg(args: list) -> None:
    """
    Run ffmpeg with given args. Raise RuntimeError on non-zero exit.
    CREATE_NO_WINDOW on Windows to suppress black console popup.
    """
    cmd = [config.FFMPEG_PATH] + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg error (exit {result.returncode}):\n{stderr_tail}")


def _run_ffprobe(args: list) -> str:
    """Run ffprobe and return stdout. Raise on non-zero exit."""
    ffprobe_path = os.path.join(
        os.path.dirname(config.FFMPEG_PATH),
        os.path.basename(config.FFMPEG_PATH).replace("ffmpeg", "ffprobe"),
    )
    cmd = [ffprobe_path] + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-500:])
    return result.stdout.decode("utf-8", errors="replace")


def get_clip_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe. Returns 0.0 on failure."""
    import json
    try:
        raw = _run_ffprobe([
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ])
        data = json.loads(raw)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def get_video_spec(video_path: str) -> dict:
    """
    Probe video for concat-relevant specs: width, height, fps, pix_fmt, sar,
    timebase denominator, audio sample_rate. Returns {} on failure.
    """
    import json
    try:
        raw = _run_ffprobe([
            "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path,
        ])
        data = json.loads(raw)
        spec = {}
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and "width" not in spec:
                spec["width"] = s.get("width")
                spec["height"] = s.get("height")
                spec["pix_fmt"] = s.get("pix_fmt", "yuv420p")
                spec["fps"] = s.get("r_frame_rate", "25/1")
                spec["sar"] = s.get("sample_aspect_ratio", "1:1")
                tb = s.get("time_base", "1/15360")
                try:
                    spec["timescale"] = int(tb.split("/")[1])
                except Exception:
                    spec["timescale"] = 15360
            elif s.get("codec_type") == "audio" and "sample_rate" not in spec:
                spec["sample_rate"] = s.get("sample_rate", "48000")
                spec["channels"] = s.get("channels", 2)
        return spec
    except Exception:
        return {}


def normalize_clip(src_path: str, output_path: str, spec: dict) -> None:
    """
    Re-encode src_path to match `spec` (from get_video_spec) so it can be safely
    concatenated with -c copy alongside clips of that spec.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    w = spec.get("width", 1920)
    h = spec.get("height", 1080)
    fps = spec.get("fps", "25/1")
    pix_fmt = spec.get("pix_fmt", "yuv420p")
    sar = (spec.get("sar") or "1:1").replace(":", "/")
    if sar in ("0/1", "N/A", ""):
        sar = "1/1"
    timescale = spec.get("timescale", 15360)
    sample_rate = str(spec.get("sample_rate", "48000"))
    channels = str(spec.get("channels", 2))

    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar={sar},fps={fps}"
    _run_ffmpeg([
        "-y",
        "-i", src_path,
        "-vf", vf,
        "-af", "aresample=async=1:first_pts=0",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", pix_fmt,
        "-video_track_timescale", str(timescale),
        "-c:a", "aac", "-ar", sample_rate, "-ac", channels,
        "-fps_mode", "cfr",
        "-shortest",
        output_path,
    ])


def _hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS to float seconds."""
    parts = hms.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _seconds_to_hms(sec: float) -> str:
    """Convert float seconds to HH:MM:SS string."""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def cut_clip(
    video_path: str,
    start: str,
    end: str,
    output_path: str,
    reencode: bool = False,
    preset: str = "veryfast",
    crf: str = "18",
) -> None:
    """
    Cut a clip from video_path between start and end (HH:MM:SS).

    reencode=False (default) — FAST: -c copy with fast seek. Clip snaps to the
      nearest keyframe at/before start. Used for B3/B4 review clips (quick to
      preview individually).
    reencode=True — ACCURATE: two-pass seek + re-encode so the clip starts
      exactly at `start` with a clean keyframe at frame 0. Required for the final
      video so subtitles stay synced and concat has no frozen frames. Slower.
      preset/crf control encode speed↔quality (veryfast/18 = quality,
      ultrafast/23 = ~4x faster, bigger files).
    Raises RuntimeError on failure.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    start_sec = _hms_to_seconds(start)
    duration_sec = _hms_to_seconds(end) - start_sec

    if reencode:
        pre_seek_sec = max(0.0, start_sec - 5)
        offset_sec = start_sec - pre_seek_sec
        _run_ffmpeg([
            "-y",
            "-ss", _seconds_to_hms(pre_seek_sec),
            "-i", video_path,
            "-ss", str(offset_sec),
            "-t", str(duration_sec),
            "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-fps_mode", "cfr",
            "-movflags", "+faststart",
            output_path,
        ])
    else:
        _run_ffmpeg([
            "-y",
            "-ss", start,
            "-i", video_path,
            "-t", str(duration_sec),
            "-c", "copy",
            output_path,
        ])


def compress_video(input_path: str, crf: str = "26", preset: str = "veryfast") -> tuple[int, int]:
    """
    Re-encode a finished video in place with efficient compression to shrink size.
    One pass over the whole video (slower preset = better compression). Audio is
    copied (already small). Returns (old_size_bytes, new_size_bytes).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    old_size = os.path.getsize(input_path)
    tmp_out = input_path + ".compress.mp4"
    _run_ffmpeg([
        "-y",
        "-i", input_path,
        "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        tmp_out,
    ])
    new_size = os.path.getsize(tmp_out)
    # Replace original with compressed (retry for Windows file lock)
    import time as _t
    for _i in range(5):
        try:
            os.replace(tmp_out, input_path)
            break
        except Exception:
            _t.sleep(0.3)
    return old_size, new_size


def merge_video(clip_paths: list[str], output_path: str, temp_dir: str) -> None:
    """
    Merge list of clip files into output_path using ffmpeg concat demuxer.
    Uses -c copy — no re-encode.
    list.txt written to temp_dir with absolute paths.
    Raises RuntimeError on failure.
    """
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Derive run_id from temp_dir name for unique list file
    run_id = os.path.basename(os.path.dirname(temp_dir))
    list_file = os.path.join(temp_dir, f"list_{run_id}.txt")

    with open(list_file, "w", encoding="utf-8") as f:
        for clip_path in clip_paths:
            abs_path = os.path.abspath(clip_path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    _run_ffmpeg([
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path,
    ])


def reset_timestamp(video_path: str, output_path: str) -> None:
    """
    Reset video timestamp to start from 00:00:00.
    Writes to output_path. Uses -c copy — no re-encode.
    Raises RuntimeError on failure.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _run_ffmpeg([
        "-y",
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ])


def delete_unused_clips(clips_dir: str, approved_clips: list[str]) -> None:
    """
    Delete all .mp4 and .srt files in clips_dir that are NOT in approved_clips.
    approved_clips: list of absolute paths to approved .mp4 files.
    Also deletes corresponding .srt for each deleted .mp4.

    Only called AFTER merge + reset_timestamp confirmed successful.
    """
    approved_set = {os.path.abspath(p) for p in approved_clips}

    for filename in os.listdir(clips_dir):
        if not filename.endswith(".mp4"):
            continue
        full_path = os.path.abspath(os.path.join(clips_dir, filename))
        if full_path not in approved_set:
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"[ffmpeg] Warning: Could not delete {full_path}: {e}")
            # Delete corresponding .srt
            srt_path = os.path.splitext(full_path)[0] + ".srt"
            if os.path.exists(srt_path):
                try:
                    os.remove(srt_path)
                except Exception as e:
                    print(f"[ffmpeg] Warning: Could not delete {srt_path}: {e}")

    # Also delete any orphan .srt files (no matching .mp4)
    for filename in os.listdir(clips_dir):
        if not filename.endswith(".srt"):
            continue
        mp4_path = os.path.join(clips_dir, os.path.splitext(filename)[0] + ".mp4")
        if not os.path.exists(mp4_path):
            try:
                os.remove(os.path.join(clips_dir, filename))
            except Exception as e:
                print(f"[ffmpeg] Warning: Could not delete orphan SRT {filename}: {e}")


def delete_temp_dir(temp_dir: str) -> None:
    """Delete temp directory and all its contents."""
    import shutil
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[ffmpeg] Warning: Could not delete temp dir {temp_dir}: {e}")
