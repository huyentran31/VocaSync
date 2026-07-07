"""
whisper_utils.py — Generate 0-based SRT from a clip's audio via OpenAI Whisper.

Used in B5 Finalize when teacher enables "Re-transcribe with Whisper".
Whisper transcribes the clip audio directly, so timestamps are 0-based and
perfectly synced to the clip — no offset math, no keyframe drift.
"""

import os

import config

_model = None


def _ensure_ffmpeg_on_path() -> None:
    """Whisper shells out to `ffmpeg` via PATH. Make sure config's ffmpeg dir is on PATH."""
    ff = getattr(config, "FFMPEG_PATH", "")
    if ff and os.path.exists(ff):
        ff_dir = os.path.dirname(ff)
        path = os.environ.get("PATH", "")
        if ff_dir and ff_dir not in path.split(os.pathsep):
            os.environ["PATH"] = ff_dir + os.pathsep + path


def _get_model():
    """Lazy-load Whisper model (cached for the process)."""
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(config.WHISPER_MODEL)
    return _model


def _fmt_ts(t: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm."""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_srt(audio_path: str, srt_path: str, language: str | None = None) -> bool:
    """
    Transcribe audio_path with Whisper and write a 0-based SRT to srt_path.
    Returns True on success, False on failure (caller can fall back to offset SRT).
    """
    if not os.path.exists(audio_path):
        return False
    try:
        _ensure_ffmpeg_on_path()
        model = _get_model()
        kwargs = {"verbose": False}
        if language:
            kwargs["language"] = language
        result = model.transcribe(audio_path, **kwargs)

        lines = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            lines.append(str(i))
            lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
            lines.append(text)
            lines.append("")

        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    except Exception as e:
        print(f"[Whisper] Transcribe failed for {audio_path}: {e}")
        return False
