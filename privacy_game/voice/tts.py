"""Text-to-speech dispatcher: Piper (real-dataset-trained) → macOS `say` fallback.

Backend selection
─────────────────
Driven by env var `PRIVACY_GAME_TTS`:
    "piper"  → require Piper; raise if unavailable
    "say"    → require macOS `say`; raise on non-macOS
    "auto"   → use Piper when available (recommended), else fall back to `say`

Default is "auto", which gives the demo zero-dep behavior on Mac while letting
researchers swap in real-dataset-trained voices with one env var + one setup
command. See tts_piper.py for the rationale and tts_setup.py for model
download.

The wire-level voice abstraction is unchanged: `CALLER_VOICE` and `AGENT_VOICE`
are the public names every caller in the codebase already imports. When Piper
is active, those abstractions map to LibriTTS-trained Piper voices instead of
opaque Apple system voices.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Voice:
    """A macOS `say` voice identifier + suggested rate (words per minute)."""

    name: str
    rate: int = 180          # 180 wpm is natural speech tempo
    description: str = ""


# Hand-picked voice pairs for clear speaker separation. Samantha is the
# default macOS "assistant" voice (high quality); Alex is a distinct male
# voice. These are pre-installed on macOS 14+.
CALLER_VOICE = Voice(name="Samantha", rate=185, description="Relying party / caller — female, conversational")
AGENT_VOICE = Voice(name="Alex", rate=180, description="Discloser / agent — male, measured")


def _ensure_say_available() -> None:
    if shutil.which("say") is None:
        raise RuntimeError(
            "macOS `say` command not found on this system. This module only "
            "supports macOS. On other platforms, swap to Piper / Coqui-TTS."
        )


def _resolve_backend() -> str:
    """Return 'piper' or 'say' based on PRIVACY_GAME_TTS + actual availability."""
    requested = os.environ.get("PRIVACY_GAME_TTS", "auto").lower()
    if requested == "say":
        return "say"
    # piper or auto: probe Piper
    try:
        from .tts_piper import is_piper_available  # local import to avoid hard dep
        piper_ready = is_piper_available()
    except Exception:
        piper_ready = False
    if requested == "piper":
        if not piper_ready:
            raise RuntimeError(
                "PRIVACY_GAME_TTS=piper but Piper is not installed or voice models are missing.\n"
                "  pip install piper-tts\n"
                "  python -m privacy_game.voice.tts_setup"
            )
        return "piper"
    # auto
    return "piper" if piper_ready else "say"


def _say_role(voice: Voice) -> str:
    """Map a Voice (which carries a macOS `say` voice name) to our caller/agent role
    so we can re-resolve to the right Piper voice."""
    return "caller" if voice.name == CALLER_VOICE.name else "agent"


def synthesize_to_file(
    text: str,
    out_path: str | Path,
    voice: Voice = AGENT_VOICE,
) -> Path:
    """Render `text` as audio and save to `out_path`. Dispatches Piper → say.

    Returns the output path. Raises on backend failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()

    if backend == "piper":
        # Piper writes WAV directly. If the caller asked for a non-.wav suffix,
        # we still produce a .wav (Piper-native) and let the caller convert.
        from .tts_piper import synthesize_to_file as piper_synth, voice_id_for
        return piper_synth(text, out_path, voice_id_for(_say_role(voice)))

    # Fallback: macOS `say`
    _ensure_say_available()
    cmd = [
        "say",
        "-v", voice.name,
        "-r", str(voice.rate),
        "-o", str(out_path),
    ]
    if out_path.suffix.lower() == ".wav":
        cmd += ["--data-format=LEI16@16000"]  # 16kHz signed-int-16 — Whisper's native rate
    subprocess.run(cmd + [text], check=True, capture_output=True)
    return out_path


def synthesize_inline(text: str, voice: Voice = AGENT_VOICE) -> bytes:
    """Render text as WAV bytes in-memory via a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        synthesize_to_file(text, tmp, voice=voice)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def speak(text: str, voice: Voice = AGENT_VOICE) -> None:
    """Play `text` through the default speaker immediately (no file saved)."""
    _ensure_say_available()
    subprocess.run(
        ["say", "-v", voice.name, "-r", str(voice.rate), text],
        check=True,
    )


def play_file(path: str | Path) -> None:
    """Play a pre-rendered audio file via `afplay`."""
    if shutil.which("afplay") is None:
        raise RuntimeError("macOS `afplay` not found.")
    subprocess.run(["afplay", str(path)], check=True)


def list_installed_voices() -> list[str]:
    """Return the list of `say` voices available on this Mac (for picking alternatives)."""
    _ensure_say_available()
    result = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True)
    lines = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if parts:
            lines.append(parts[0])
    return lines


if __name__ == "__main__":
    # Smoke test — render both voices
    out_dir = Path("/tmp/privacy_game_tts_test")
    out_dir.mkdir(exist_ok=True)
    print("Rendering caller voice...")
    synthesize_to_file(
        "Hello, this is your pharmacy calling. What medication are you refilling today?",
        out_dir / "caller.wav",
        voice=CALLER_VOICE,
    )
    print("Rendering agent voice...")
    synthesize_to_file(
        "I'd rather not say the drug name. It's an oral antidiabetic.",
        out_dir / "agent.wav",
        voice=AGENT_VOICE,
    )
    print(f"Wrote {out_dir}/caller.wav and {out_dir}/agent.wav")
    print(f"Available voices on this Mac: {', '.join(list_installed_voices()[:12])}...")
