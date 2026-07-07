"""
ai_client.py — AI API client for AI Extraction mode.

Supports 2 providers via AI_PROVIDER in config.py:
  "gemini"            — Google Gemini native endpoint
  "openai_compatible" — OpenRouter, RunningHub, etc.

Functions:
  get_timestamps       — B2: send SRT to AI, receive JSON timestamp list
  send_needs_revision  — B4: send needs_revision clips to AI for re-selection
  tag_already_used     — B5*: wrap approved segments with [ALREADY USED] tags
"""

import json
import os
import re
import sys
import time

import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Load the active system prompt.
    Reads config/active_prompt.txt (a pointer to a file in config/prompts/);
    falls back to config/system_prompt.txt for backward compatibility."""
    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    prompt_path = os.path.join(cfg_dir, "system_prompt.txt")  # legacy fallback
    try:
        pointer = os.path.join(cfg_dir, "active_prompt.txt")
        if os.path.exists(pointer):
            with open(pointer, "r", encoding="utf-8") as f:
                name = f.read().strip()
            cand = os.path.join(cfg_dir, "prompts", name)
            if name and os.path.exists(cand):
                prompt_path = cand
    except Exception:
        pass
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _call_gemini(prompt: str, system_prompt: str, model: str | None = None) -> str:
    """Call Gemini native endpoint. Returns raw text response.

    `model` overrides config.AI_MODEL for this single call (model routing, Day-5).
    """
    model = model or config.AI_MODEL
    # API key goes in a HEADER, never the URL — a requests exception embeds the full
    # URL, which would leak the key into logs/trajectory/UI error banners.
    url = f"{config.AI_ENDPOINT}{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.AI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=config.API_TIMEOUT)

    if response.status_code == 429:
        raise RuntimeError("Rate limit (429) — slow down or increase SLEEP_BETWEEN_BATCHES")
    if response.status_code != 200:
        error_msg = response.json().get("error", {}).get("message", response.text[:300])
        raise RuntimeError(f"Gemini API error (HTTP {response.status_code}): {error_msg}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates. Full response: {str(data)[:500]}")

    return candidates[0]["content"]["parts"][0]["text"]


def _call_openai_compatible(prompt: str, system_prompt: str, model: str | None = None) -> str:
    """Call OpenAI-compatible endpoint. Returns raw text response.

    `model` overrides config.AI_MODEL for this single call (model routing, Day-5).
    """
    url = config.AI_ENDPOINT.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.AI_API_KEY}",
    }
    payload = {
        "model": model or config.AI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 8192,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=config.API_TIMEOUT)

    if response.status_code == 429:
        raise RuntimeError("Rate limit (429) — slow down or increase SLEEP_BETWEEN_BATCHES")
    if response.status_code != 200:
        raise RuntimeError(f"API error (HTTP {response.status_code}): {response.text[:300]}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_ai(prompt: str, system_prompt: str, model: str | None = None) -> str:
    """Dispatch to correct provider. Retries up to 3 times on 429 with exponential backoff.

    `model` (optional) overrides config.AI_MODEL for this call only — used for model
    routing (Day-5), e.g. extract_vocab's self-correct fix call uses the strong model.
    Default None keeps the original single-model behaviour for every existing caller.
    """
    # S18 askfix (owner decision): the old fixed 30s base (30/60/120s) made every throttled
    # question stall for minutes — far harsher than the 15-RPM window needs (a slot frees
    # within seconds). New policy per owner: FLAT 10s per retry; only after 5 throttled
    # attempts escalate to 20s. More, shorter retries beat few, long ones for a per-minute
    # quota. Env knob AI_429_BASE_SLEEP rescales both steps without a code change.
    max_retries = 8
    base_sleep = float(os.getenv("AI_429_BASE_SLEEP", "10"))

    for attempt in range(max_retries):
        try:
            if config.AI_PROVIDER == "gemini":
                return _call_gemini(prompt, system_prompt, model)
            elif config.AI_PROVIDER == "openai_compatible":
                return _call_openai_compatible(prompt, system_prompt, model)
            else:
                raise ValueError(f"Unknown AI_PROVIDER: '{config.AI_PROVIDER}' — must be 'gemini' or 'openai_compatible'")
        except RuntimeError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                sleep_sec = base_sleep * (1 if attempt < 5 else 2)   # 10s ×5, then 20s
                print(f"[ai_client] Rate limit (429) — retrying in {sleep_sec}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_sec)
            else:
                raise


# ---------------------------------------------------------------------------
# JSON parsing — 3 layers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"video", "start", "end", "ai_note", "example_sub", "language_item", "language_tag", "original_word"}


def _normalize_hms(ts: str) -> str:
    """Strip milliseconds from AI timestamp variants → HH:MM:SS.
    Handles: HH:MM:SS:mmm  HH:MM:SS,mmm  HH:MM:SS.mmm
    """
    ts = ts.strip()
    m = re.match(r"(\d{2}:\d{2}:\d{2})[,.:]\d+", ts)
    return m.group(1) if m else ts


def _parse_ai_response(raw_text: str, context: str = "", require_ai_note: bool = True) -> list[dict]:
    """
    Parse AI response through 3 layers:
      Layer 1: Strip markdown fence if present
      Layer 2: Parse JSON — log position on error
      Layer 3: Validate each item has required fields

    require_ai_note: if True (default, B2), skip items with empty ai_note.
                     Set False for B4 needs_revision (ai_note optional).

    Returns list of valid items. Invalid items are skipped with warning.
    Raises RuntimeError if JSON parse fails entirely.
    """
    # Layer 1: Strip markdown fence
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Layer 2: Parse JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"JSON parse error at line {e.lineno}, col {e.colno}: {e.msg}\n"
            f"Context: {context}\n"
            f"Raw response (first 1000 chars):\n{raw_text[:1000]}"
        )

    # Unwrap if AI returned {"results": [...]} or similar
    if isinstance(parsed, dict):
        for key in parsed:
            if isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected JSON array, got {type(parsed).__name__}. Context: {context}")

    # Layer 3: Validate each item
    valid_items = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            print(f"[ai_client] Warning: Item {i} is not a dict — skipping")
            continue
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            print(f"[ai_client] Warning: Item {i} missing fields {missing} — skipping")
            continue
        if require_ai_note and not item.get("ai_note"):
            print(f"[ai_client] Warning: Item {i} has empty ai_note — skipping")
            continue
        item["start"] = _normalize_hms(str(item.get("start", "")))
        item["end"] = _normalize_hms(str(item.get("end", "")))
        valid_items.append(item)

    return valid_items


# ---------------------------------------------------------------------------
# B2 — get_timestamps
# ---------------------------------------------------------------------------

def get_timestamps(
    pairs: list[dict],
    topic: str,
    audience: str,
    additional_requirements: str = "",
    video_source_info: str = "",
) -> list[dict]:
    """
    B2 AI Extraction: Send SRT content to AI in batches.
    Each pair: {"video": abs_path, "srt": abs_path}

    Returns list of dicts matching JSON schema (8 fields).
    Failed videos: logged, skipped, do not crash pipeline.
    """
    system_prompt = _load_system_prompt()

    batch_size = config.SO_VIDEO_PER_BATCH if config.SO_VIDEO_PER_BATCH > 0 else len(pairs)
    all_results = []

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        is_last_batch = (batch_start + batch_size) >= len(pairs)

        batch_video_filenames = [os.path.basename(p["video"]) for p in batch]

        prompt_parts = [
            f"Topic: {topic}",
            f"Target audience: {audience}",
        ]
        if additional_requirements:
            prompt_parts.append(f"Additional requirements: {additional_requirements}")
        if video_source_info:
            prompt_parts.append(f"Video source info: {video_source_info}")
        prompt_parts.append("")

        total_words = 0
        for pair in batch:
            video_filename = os.path.basename(pair["video"])
            try:
                with open(pair["srt"], "r", encoding="utf-8") as f:
                    srt_content = f.read()
            except Exception as e:
                print(f"[ai_client] Warning: Cannot read SRT for {video_filename}: {e}")
                continue
            total_words += len(srt_content.split())
            prompt_parts.append(f"--- VIDEO: {video_filename} ---")
            prompt_parts.append(srt_content)
            prompt_parts.append("")

        if total_words > config.SUB_WARNING_THRESHOLD:
            print(f"[ai_client] Warning: Batch sub word count {total_words} exceeds threshold {config.SUB_WARNING_THRESHOLD}")

        prompt = "\n".join(prompt_parts)
        # F2: check full prompt word count against WARNING_THRESHOLD
        _wc_f2 = len(prompt.split())
        _wt_f2 = getattr(config, "WARNING_THRESHOLD", 50000)
        if _wc_f2 > _wt_f2:
            print(f"[ai_client] Warning: Full prompt word count {_wc_f2} exceeds WARNING_THRESHOLD {_wt_f2}")
        context = f"B2 batch {batch_start // batch_size + 1}"
        try:
            raw = call_ai(prompt, system_prompt)
            items = _parse_ai_response(raw, context=context)

            # Validate video field matches one of the batch filenames
            valid = []
            for item in items:
                v = item.get("video", "")
                if v not in batch_video_filenames:
                    print(f"[ai_client] Warning: Item has unknown video '{v}' — skipping")
                    continue
                valid.append(item)

            all_results.extend(valid)
            print(f"[ai_client] {context}: received {len(valid)} segments")
        except Exception as e:
            print(f"[ai_client] Error in {context}: {e}")

        if not is_last_batch:
            print(f"[ai_client] Sleeping {config.SLEEP_BETWEEN_BATCHES}s before next batch...")
            time.sleep(config.SLEEP_BETWEEN_BATCHES)

    return all_results


# ---------------------------------------------------------------------------
# B4 — send_needs_revision
# ---------------------------------------------------------------------------

def send_needs_revision(
    revision_clips: list[dict],
    topic: str,
    audience: str,
    additional_requirements: str = "",
    video_source_info: str = "",
    clips_per_batch: int = 0,
) -> list[dict]:
    """
    B4 Phase 2: Send needs_revision clips back to AI.

    revision_clips: list of dicts with keys:
      global_index, video, start, end, ai_note, comment, sub_lines,
      clip_srt_path (path to clip .srt file),
      source_srt_path (path to full source SRT for context expansion)

    Groups clips by video. Chunks per SO_CLIP_NEEDS_REVISION within each video.
    JSON response must include global_index for mapping.

    Returns list of new segment dicts (with global_index).
    """
    system_prompt = _load_system_prompt()
    batch_size = clips_per_batch if clips_per_batch > 0 else len(revision_clips)

    # Group by video
    by_video: dict[str, list[dict]] = {}
    for clip in revision_clips:
        video = clip["video"]
        if video not in by_video:
            by_video[video] = []
        by_video[video].append(clip)

    all_results = []

    for video_name, clips in by_video.items():
        for batch_start in range(0, len(clips), batch_size):
            batch = clips[batch_start : batch_start + batch_size]
            is_last = (batch_start + batch_size) >= len(clips)

            prompt_parts = [
                f"Topic: {topic}",
                f"Target audience: {audience}",
            ]
            if additional_requirements:
                prompt_parts.append(f"Additional requirements: {additional_requirements}")
            if video_source_info:
                prompt_parts.append(f"Video source info: {video_source_info}")
            prompt_parts.append("")
            prompt_parts.append("This is a revision request. For each clip below, find a DIFFERENT segment as replacement based on teacher feedback.")
            prompt_parts.append("Your JSON response must include 'global_index' to identify which clip each result replaces.")
            prompt_parts.append("")

            for clip in batch:
                clip_srt = ""
                if clip.get("clip_srt_path") and os.path.exists(clip["clip_srt_path"]):
                    with open(clip["clip_srt_path"], "r", encoding="utf-8") as f:
                        clip_srt = f.read()

                context_sub = ""
                if clip.get("source_srt_path") and os.path.exists(clip["source_srt_path"]):
                    from file_utils import read_sub, expand_sub_context
                    source_srt = read_sub(clip["source_srt_path"])
                    exp_start, exp_end = expand_sub_context(
                        source_srt,
                        clip["start"],
                        int(clip.get("sub_lines", config.DEFAULT_SUB_LINES)),
                    )
                    # Extract context lines
                    from file_utils import create_clip_subtitle
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w", encoding="utf-8") as tmp:
                        tmp_path = tmp.name
                    try:
                        create_clip_subtitle(source_srt, exp_start, exp_end, tmp_path)
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            context_sub = f.read()
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                block = []
                block.append(f"CLIP {clip['global_index']} (rejected: {clip['start']} → {clip['end']})")
                if clip.get("ai_note"):
                    block.append(f"Previous AI reason: {clip['ai_note']}")
                if clip.get("comment"):
                    block.append(f"Teacher feedback: {clip['comment']}")
                if clip_srt:
                    block.append("Rejected clip subtitles:")
                    block.append(clip_srt)
                if context_sub:
                    block.append("Context subtitles for replacement search:")
                    block.append(context_sub)
                block.append("")

                # Use .replace() not .format()
                prompt_parts.append("\n".join(block))

            prompt = "\n".join(prompt_parts)
            context = f"B4 needs_revision video={video_name} batch={batch_start//batch_size + 1}"

            try:
                raw = call_ai(prompt, system_prompt)
                items = _parse_ai_response(raw, context=context, require_ai_note=False)

                # Validate global_index present
                valid = []
                for item in items:
                    if "global_index" not in item:
                        print(f"[ai_client] Warning: Item missing global_index in {context} — skipping")
                        continue
                    valid.append(item)

                all_results.extend(valid)
                print(f"[ai_client] {context}: received {len(valid)} revised segments")
            except Exception as e:
                print(f"[ai_client] Error in {context}: {e}")

            if not is_last:
                time.sleep(config.SLEEP_BETWEEN_BATCHES)

    return all_results


# ---------------------------------------------------------------------------
# B5* — tag_already_used (wrapper — actual tagging in file_utils)
# ---------------------------------------------------------------------------

def tag_already_used(approved_segments: list[dict], srt_content: str) -> str:
    """
    Wrapper for B5* Find More Examples.
    Delegates to file_utils.filter_sub_excluding_approved.
    """
    from file_utils import filter_sub_excluding_approved
    return filter_sub_excluding_approved(srt_content, approved_segments)
