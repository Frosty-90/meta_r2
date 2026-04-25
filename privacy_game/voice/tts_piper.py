"""Piper TTS backend — voice models trained on REAL human recordings.

Why this module exists
──────────────────────
The default macOS `say` backend ships great-sounding voices, but it's macOS-only
and the underlying training data is opaque — Apple doesn't disclose the corpora.
For honest "real dataset" work, we want a TTS whose voices come from publicly-
documented, real human speech corpora.

Piper (https://github.com/rhasspy/piper) is exactly that:
    - Open-source neural TTS, ONNX-runtime, runs on CPU
    - Voice models are trained on:
        * LibriTTS         (Zen et al. 2019) — derived from LibriSpeech audiobooks
        * VCTK             (Yamagishi et al. 2019) — 110 native English speakers
        * Mozilla Common Voice — crowd-sourced multilingual speech corpus
    - Each voice's training corpus + license are published in the model card on
      https://huggingface.co/rhasspy/piper-voices
    - Sub-real-time inference on Apple Silicon (M-series) CPU, no GPU required

Activation
──────────
Piper is OPT-IN. Install + download voice models once, then either:
    1. Set `PRIVACY_GAME_TTS=piper`         (forces Piper, errors if unavailable)
    2. Leave `PRIVACY_GAME_TTS=auto` (default) — uses Piper when present,
       falls back to macOS `say` when not.

Setup (one-time):
    pip install piper-tts
    python -m privacy_game.voice.tts_setup     # downloads ~150MB of voices

Hand-picked voices (defaults)
─────────────────────────────
    Caller (Relying Party):  en_US-amy-medium    — female, LibriTTS-trained
    Agent  (Discloser):      en_US-ryan-medium   — male,   LibriTTS-trained

Both are MIT/CC-BY licensed. Override via PRIVACY_GAME_PIPER_CALLER_VOICE /
PRIVACY_GAME_PIPER_AGENT_VOICE if you want to swap in VCTK or Common Voice
speakers (see tts_setup.py for the catalogue).
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Configuration

PIPER_MODELS_DIR = Path(
    os.environ.get(
        "PRIVACY_GAME_PIPER_MODELS",
        str(Path.home() / ".cache" / "privacy_game" / "piper"),
    )
)

# Default voice IDs. Overridable via env vars for evaluation across speakers.
DEFAULT_CALLER_VOICE = os.environ.get("PRIVACY_GAME_PIPER_CALLER_VOICE", "en_US-amy-medium")
DEFAULT_AGENT_VOICE = os.environ.get("PRIVACY_GAME_PIPER_AGENT_VOICE", "en_US-ryan-medium")


# ──────────────────────────────────────────────────────────────────────────────
# Availability + caching

_voice_cache: dict[str, "PiperVoice"] = {}  # type: ignore[name-defined]


def is_piper_available(caller_voice: Optional[str] = None, agent_voice: Optional[str] = None) -> bool:
    """True if `piper-tts` is importable AND both voice models are on disk."""
    try:
        import piper.voice  # type: ignore[import-not-found] # noqa: F401
    except ImportError:
        return False
    cv = caller_voice or DEFAULT_CALLER_VOICE
    av = agent_voice or DEFAULT_AGENT_VOICE
    needed = []
    for vid in (cv, av):
        needed.append(PIPER_MODELS_DIR / f"{vid}.onnx")
        needed.append(PIPER_MODELS_DIR / f"{vid}.onnx.json")
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


import re as _re
_VOICE_ID_PATTERN = _re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_voice_id(voice_id: str) -> None:
    """Reject voice_ids that could escape PIPER_MODELS_DIR via path tricks.

    Found in red-team v2: `voice_id="../../etc/passwd"` was interpolated into
    a Path and reached the filesystem (raised FileNotFoundError but exposed
    the interpolation behavior). Here we whitelist the legal alphabet so any
    path-shaped input gets rejected at the seam.
    """
    if not isinstance(voice_id, str) or not _VOICE_ID_PATTERN.match(voice_id):
        raise ValueError(
            f"invalid voice_id {voice_id!r}; expected [a-zA-Z0-9_-]{{1,64}}"
        )


def _load_voice(voice_id: str):
    """Load + memoize a PiperVoice. Heavy ONNX cold-start; warm via Whisper-style."""
    _validate_voice_id(voice_id)
    if voice_id in _voice_cache:
        return _voice_cache[voice_id]
    onnx = PIPER_MODELS_DIR / f"{voice_id}.onnx"
    config = PIPER_MODELS_DIR / f"{voice_id}.onnx.json"
    if not onnx.exists() or not config.exists():
        raise FileNotFoundError(
            f"Piper voice model {voice_id!r} not found at {PIPER_MODELS_DIR}.\n"
            f"Run: python -m privacy_game.voice.tts_setup {voice_id}"
        )
    from piper.voice import PiperVoice  # type: ignore[import-not-found] # optional runtime dep — gated by is_piper_available()
    voice = PiperVoice.load(str(onnx), str(config))
    _voice_cache[voice_id] = voice
    return voice


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis API — mirrors tts.py shape so it's drop-in

def synthesize_to_file(text: str, out_path: Path, voice_id: str) -> Path:
    """Render `text` as a 16-bit PCM WAV at the voice's native sample rate.

    Args:
        text:     Input string. Piper handles punctuation + sentence breaks.
        out_path: Output WAV path. Parent dirs created if missing.
        voice_id: e.g. "en_US-amy-medium". Must be on disk under PIPER_MODELS_DIR.

    Returns:
        Path to the written WAV.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    voice = _load_voice(voice_id)
    with wave.open(str(out_path), "wb") as wf:
        # Piper writes 22.05kHz mono int16 by default. Whisper handles any rate
        # via internal resampling, so no need to force 16kHz here.
        voice.synthesize(text, wf)
    return out_path


def voice_id_for(role: str) -> str:
    """Resolve our caller/agent role → concrete Piper voice id."""
    if role == "caller":
        return DEFAULT_CALLER_VOICE
    if role == "agent":
        return DEFAULT_AGENT_VOICE
    raise ValueError(f"role must be 'caller' or 'agent', got {role!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test

if __name__ == "__main__":
    if not is_piper_available():
        print("Piper not available. Install + download voices:")
        print("  pip install piper-tts")
        print("  python -m privacy_game.voice.tts_setup")
        raise SystemExit(1)

    out = Path("/tmp/piper_smoke")
    out.mkdir(exist_ok=True)
    for role in ("caller", "agent"):
        vid = voice_id_for(role)
        path = out / f"{role}.wav"
        synthesize_to_file(
            f"Hello from Piper, this is the {role} voice. Trained on real LibriTTS speakers.",
            path,
            vid,
        )
        print(f"  ✅ {role:6s}  voice={vid}  →  {path}  ({path.stat().st_size} bytes)")
