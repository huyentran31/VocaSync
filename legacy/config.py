"""
config.py — shim that the bundled legacy modules import (`import config`).

The original AI-Teaching tool shipped a hand-edited config.py; this repo keeps
secrets OUT of code (Day-4), so values come from environment / `.env` with safe
defaults. Loading `.env` is best-effort: we parse it ourselves (no extra dep) so
the shim works even before python-dotenv is installed.

Anything the legacy whisper/ffmpeg/ai_client/xlsx modules reference at import or
call time must have a name here. Add new keys to `.env.example` when you add one.
"""

from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# .env loader (tiny, dependency-free) — does NOT overwrite real env vars.
# --------------------------------------------------------------------------- #

def _load_dotenv() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


_load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- #
# AI provider (Day-5 model routing). Provider = legacy Gemini / OpenAI-compatible.
# Keys NEVER hardcoded — only read from env. See .env.example.
# --------------------------------------------------------------------------- #

AI_PROVIDER = _get("AI_PROVIDER", "gemini")  # "gemini" | "openai_compatible"

# Gemini native endpoint (note trailing slash + model appended by ai_client).
AI_ENDPOINT = _get("AI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models/")
# Default workhorse = flash-lite (most free-tier quota). NOTE: do NOT fall back to
# MODEL_STRONG here — MODEL_STRONG may be a Claude id (.env.example) which a Gemini
# endpoint rejects with 404. Keep AI_MODEL a Gemini id when AI_PROVIDER=gemini.
AI_MODEL = _get("AI_MODEL", "gemini-3.1-flash-lite")
# Strong model used SPARINGLY (e.g. extract_vocab's self-correct fix call).
# Verified id (ListModels): gemini-3.1-flash-lite (RPM 15 / RPD 500) — same model
# everywhere keeps the free-tier quota predictable and avoids a 404 on a wrong id.
AI_MODEL_STRONG = _get("AI_MODEL_STRONG", "gemini-3.1-flash-lite")
AI_MODEL_CHEAP = _get("MODEL_CHEAP", "gemini-3.1-flash-lite")
AI_API_KEY = _get("GEMINI_API_KEY") or _get("AI_API_KEY") or _get("OPENAI_API_KEY")

API_TIMEOUT = int(_get("API_TIMEOUT", "120"))
SLEEP_BETWEEN_BATCHES = int(_get("SLEEP_BETWEEN_BATCHES", "5"))

# Batching / warning thresholds used by legacy ai_client (kept for compatibility).
SO_VIDEO_PER_BATCH = int(_get("SO_VIDEO_PER_BATCH", "0"))      # 0 => all in one batch
SUB_WARNING_THRESHOLD = int(_get("SUB_WARNING_THRESHOLD", "30000"))
WARNING_THRESHOLD = int(_get("WARNING_THRESHOLD", "50000"))
SO_CLIP_NEEDS_REVISION = int(_get("SO_CLIP_NEEDS_REVISION", "0"))
DEFAULT_SUB_LINES = int(_get("DEFAULT_SUB_LINES", "2"))


# --------------------------------------------------------------------------- #
# Media tooling. ffmpeg/whisper are SYSTEM deps — a missing binary is a
# system-error (AGENTS.md §4), surfaced by the tools that need it.
# --------------------------------------------------------------------------- #

FFMPEG_PATH = _get("FFMPEG_PATH", "ffmpeg")   # on PATH by default
WHISPER_MODEL = _get("WHISPER_MODEL", "base")

# S18 P0-1: seconds of padding added to EACH edge of an Anki audio/screenshot clip so a
# short cue isn't clipped mid-word or into silence. 0.2 is deliberate — dense/fast film
# dialogue makes a larger pad bleed into the neighbouring line. Bump to 0.3 if onsets
# sound cut. Lower edge is clamped to 0; upper edge only when the media duration is known.
ANKI_CLIP_PAD = float(_get("ANKI_CLIP_PAD", "0.2"))


# --------------------------------------------------------------------------- #
# ConceptNet (Day-2 interoperability) — supplements WordNet for the "life-context"
# layer (used_for / has_context) and fills sparse/OOV part_of. Deterministic source
# (REST API returns facts; the LLM only PICKS/VETS, never invents). Online by default;
# swap to conceptnet-lite for offline without touching the schema.
# --------------------------------------------------------------------------- #

CONCEPTNET_ENDPOINT = _get("CONCEPTNET_ENDPOINT", "https://api.conceptnet.io")
# Precision filter: weight 1.0 == single source; >=1.5 keeps multi-source agreement.
# Start low and tighten after seeing real data (do NOT hardcode 2.0 — too aggressive).
CONCEPTNET_MIN_WEIGHT = float(_get("CONCEPTNET_MIN_WEIGHT", "1.5"))
CONCEPTNET_TIMEOUT = int(_get("CONCEPTNET_TIMEOUT", "10"))
# B1: the deterministic Mine pipeline queries ConceptNet for EVERY term (rich graph).
# Set CONCEPTNET_PER_TERM=0 to save calls (only the agent path will then use it).
CONCEPTNET_PER_TERM = _get("CONCEPTNET_PER_TERM", "1") not in ("0", "false", "False", "")
CONCEPTNET_MAX_EDGES = int(_get("CONCEPTNET_MAX_EDGES", "30"))   # cap per lookup (anti graph-explosion)


def has_ai_key() -> bool:
    """Cheap check used by AI tools to raise a clear system-error early."""
    return bool(AI_API_KEY)
