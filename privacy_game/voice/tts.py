"""Text-to-speech wrapper using macOS `say`.

Why macOS `say`:
    - Zero external dependencies (system command)
    - High-quality neural voices on M3 (Samantha, Alex, Karen, Daniel, etc.)
    - Distinct voices for caller vs agent gives the demo a proper phone-call feel
    - Fast enough for real-time demo capture

For cross-platform or higher fidelity, swap in Piper / Kokoro / Coqui-TTS.
"""

from __future__ import annotations

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


def synthesize_to_file(
    text: str,
    out_path: str | Path,
    voice: Voice = AGENT_VOICE,
) -> Path:
    """Render `text` as audio and save to `out_path` (AIFF or WAV based on suffix).

    Returns the output path. Raises CalledProcessError on failure.
    """
    _ensure_say_available()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # macOS `say` accepts -o <file> and infers format from suffix.
    # --data-format=LEF32@22050 yields 22.05kHz float32 WAV (good for Whisper + portable).
    cmd = [
        "say",
        "-v", voice.name,
        "-r", str(voice.rate),
        "-o", str(out_path),
    ]
    if out_path.suffix.lower() == ".wav":
        cmd += ["--data-format=LEI16@16000"]  # 16kHz signed-int-16 — Whisper's native rate

    subprocess.run(
        cmd + [text],
        check=True,
        capture_output=True,
    )
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
