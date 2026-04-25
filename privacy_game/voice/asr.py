"""Speech-to-text wrapper using faster-whisper.

faster-whisper runs Whisper models via CTranslate2 — substantially faster and
lower-memory than stock OpenAI whisper. Runs on CPU on M3 at reasonable speed
(base.en transcribes 10s of speech in ~1-2s).

Models to choose from (English-only, CPU-friendly on M3):
    tiny.en   39MB    fastest, lower accuracy
    base.en  140MB    recommended default — good accuracy, fast
    small.en 460MB    higher accuracy, slower
    medium.en 1.5GB   diminishing returns on clean speech

First invocation downloads the model to ~/.cache/huggingface/hub/.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptSegment:
    start: float    # seconds
    end: float
    text: str


@dataclass
class Transcript:
    text: str                        # full concatenated transcript
    segments: list[TranscriptSegment]
    language: str
    duration: float                  # audio duration in seconds


class ASR:
    """Lazy-loaded Whisper wrapper.

    Instantiate once; reuse across many calls. Model download happens on first
    transcribe() call.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",   # int8 on CPU is fast & memory-efficient
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Suppress HF download progress noise in non-interactive runs
                    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                    from faster_whisper import WhisperModel  # lazy import
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
        beam_size: int = 5,
    ) -> Transcript:
        self._ensure_loaded()
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            # vad_filter=False by default; enabling VAD tends to drop short/quiet turns
            # which is bad for our use case where agents sometimes say very short things.
        )
        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
        ]
        full_text = " ".join(s.text for s in segments).strip()
        return Transcript(
            text=full_text,
            segments=segments,
            language=info.language or language,
            duration=float(info.duration or 0.0),
        )


# Module-level singleton for convenience (loaded lazily)
_default_asr: Optional[ASR] = None


def default_asr() -> ASR:
    global _default_asr
    if _default_asr is None:
        _default_asr = ASR()
    return _default_asr


def transcribe(audio_path: str | Path) -> str:
    """Convenience: transcribe `audio_path` and return just the text."""
    return default_asr().transcribe(audio_path).text


if __name__ == "__main__":
    # Round-trip test: our own TTS output → ASR → compare
    from .tts import synthesize_to_file, CALLER_VOICE, AGENT_VOICE

    out_dir = Path("/tmp/privacy_game_asr_test")
    out_dir.mkdir(exist_ok=True)

    test_pairs = [
        (CALLER_VOICE, "Hello, this is your pharmacy calling. What medication are you refilling today?"),
        (AGENT_VOICE, "I'd rather not say the drug name. It's an oral antidiabetic."),
        (CALLER_VOICE, "What is your zip code for regional pricing?"),
        (AGENT_VOICE, "I'm in the 021 area."),
        (AGENT_VOICE, "I'd rather not share that information."),
    ]

    asr = default_asr()
    print(f"Loading Whisper {asr.model_size} (first run will download ~140MB)...")

    for i, (voice, text) in enumerate(test_pairs):
        wav = out_dir / f"test_{i:02d}.wav"
        synthesize_to_file(text, wav, voice=voice)
        tr = asr.transcribe(wav)
        match = "✓" if text.lower().replace(".", "").replace(",", "").strip() in tr.text.lower() else "✗"
        print(f"{match} [{voice.name}] original : {text!r}")
        print(f"   transcribed: {tr.text!r}")
        print(f"   duration: {tr.duration:.2f}s")
        print()
