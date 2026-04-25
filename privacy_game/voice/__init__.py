"""Voice extension for the Contextual-Integrity Disclosure Game.

Research framing: we train on text because text is the verifiable unit, but we
deploy to voice because that's where real humans and LLMs over-share in the
wild (phone calls, voice assistants, smart-speaker dialogues). If the text-
trained policy generalizes to voice, that's evidence the skill is about
information control, not about text-specific syntax.

Modules:
    asr.py          — Whisper-based speech-to-text
    tts.py          — macOS `say`-based text-to-speech (+ playback)
    voice_shell.py  — end-to-end voice-layer shell over an episode
    audio_eval.py   — held-out audio eval: base vs trained, same policies,
                      voice modality, measure leak rate
    demo_render.py  — pre-render demo scenarios as MP3/WAV for video capture
"""
