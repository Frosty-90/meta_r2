"""Download Piper voice models for the privacy game demo.

Voices come from `huggingface.co/rhasspy/piper-voices`. Each model card lists
the training corpus and license. We pre-select voices trained on REAL human
speech (LibriTTS, VCTK, or Mozilla Common Voice).

Usage:
    python -m privacy_game.voice.tts_setup                  # download defaults
    python -m privacy_game.voice.tts_setup --list           # show catalogue
    python -m privacy_game.voice.tts_setup en_US-amy-medium en_US-ryan-medium  # specific voices

Default download size: ~150MB (two voices × ~75MB).
Cache dir: ~/.cache/privacy_game/piper/  (override with PRIVACY_GAME_PIPER_MODELS)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from .tts_piper import PIPER_MODELS_DIR, DEFAULT_CALLER_VOICE, DEFAULT_AGENT_VOICE


# ──────────────────────────────────────────────────────────────────────────────
# Voice catalogue. All voice models are published at huggingface.co/rhasspy/piper-voices
# under MIT / CC-BY licenses. Training-corpus attribution is the whole point of
# this module — every voice here comes from a publicly-documented real-speech dataset.

# URL template: https://huggingface.co/rhasspy/piper-voices/resolve/main/<lang>/<lang_locale>/<name>/<quality>/<voice_id>.onnx
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

VOICES: dict[str, dict] = {
    # ── English / US ─────────────────────────────────────────────────────────
    "en_US-amy-medium": {
        "lang": "en",
        "locale": "en_US",
        "name": "amy",
        "quality": "medium",
        "gender": "female",
        "dataset": "LibriTTS",
        "license": "MIT",
        "note": "Default caller voice — LibriTTS-trained, real audiobook reader.",
    },
    "en_US-ryan-medium": {
        "lang": "en",
        "locale": "en_US",
        "name": "ryan",
        "quality": "medium",
        "gender": "male",
        "dataset": "LibriTTS",
        "license": "MIT",
        "note": "Default agent voice — LibriTTS-trained, real audiobook reader.",
    },
    "en_US-lessac-medium": {
        "lang": "en",
        "locale": "en_US",
        "name": "lessac",
        "quality": "medium",
        "gender": "female",
        "dataset": "LJSpeech / Lessac",
        "license": "Public domain (LJSpeech)",
        "note": "Clear, neutral; LJSpeech corpus.",
    },
    "en_US-libritts-high": {
        "lang": "en",
        "locale": "en_US",
        "name": "libritts",
        "quality": "high",
        "gender": "multi",
        "dataset": "LibriTTS",
        "license": "CC BY 4.0",
        "note": "Multi-speaker (904 voices) — pick speaker via speaker_id at synthesis time.",
    },
    # ── English / UK ─────────────────────────────────────────────────────────
    "en_GB-jenny_dioco-medium": {
        "lang": "en",
        "locale": "en_GB",
        "name": "jenny_dioco",
        "quality": "medium",
        "gender": "female",
        "dataset": "Dioco-Jenny",
        "license": "MIT",
        "note": "British female; useful for accent-robustness eval.",
    },
    "en_GB-alan-medium": {
        "lang": "en",
        "locale": "en_GB",
        "name": "alan",
        "quality": "medium",
        "gender": "male",
        "dataset": "Alan corpus",
        "license": "MIT",
        "note": "British male; pair with jenny for UK demo.",
    },
}


def voice_url(voice_id: str, ext: str) -> str:
    """Build the HF raw-file URL for a model file."""
    if voice_id not in VOICES:
        raise KeyError(f"Unknown voice {voice_id!r}. See --list.")
    v = VOICES[voice_id]
    fname = f"{voice_id}.onnx" + (".json" if ext == "json" else "")
    return f"{HF_BASE}/{v['lang']}/{v['locale']}/{v['name']}/{v['quality']}/{fname}"


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _download_with_progress(url: str, out_path: Path) -> None:
    """Stream a download with a single-line progress bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    print(f"  ⤓ {url.split('/')[-1]}")
    with urllib.request.urlopen(url, timeout=30) as r:
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 64
        with tmp.open("wb") as fh:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                downloaded += len(buf)
                if total > 0:
                    pct = 100 * downloaded / total
                    bar = "█" * int(pct / 4) + "·" * (25 - int(pct / 4))
                    print(f"      [{bar}] {pct:5.1f}% ({_human_size(downloaded)}/{_human_size(total)})",
                          end="\r", flush=True)
        print()  # newline after progress bar
    shutil.move(str(tmp), str(out_path))


def download_voice(voice_id: str, dest_dir: Path) -> bool:
    """Download both onnx + onnx.json for a voice. Returns True if any new file was fetched."""
    onnx_path = dest_dir / f"{voice_id}.onnx"
    json_path = dest_dir / f"{voice_id}.onnx.json"
    info = VOICES[voice_id]
    print(f"\n→ {voice_id}  ({info['gender']}, {info['lang']}, dataset: {info['dataset']}, license: {info['license']})")
    print(f"  {info['note']}")

    fetched_any = False
    for path, ext in [(json_path, "json"), (onnx_path, "onnx")]:
        if path.exists() and path.stat().st_size > 0:
            print(f"  ✓ {path.name} already present ({_human_size(path.stat().st_size)})")
            continue
        url = voice_url(voice_id, ext)
        try:
            _download_with_progress(url, path)
            fetched_any = True
        except Exception as e:
            print(f"  ❌ failed: {e}")
            return False
    return fetched_any or True


def list_voices() -> None:
    print(f"{'voice id':<28} {'lang':<7} {'gender':<8} {'dataset':<24} note")
    print("-" * 100)
    for vid, v in VOICES.items():
        print(f"{vid:<28} {v['lang']:<7} {v['gender']:<8} {v['dataset']:<24} {v['note']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voices", nargs="*",
                        help="Voice IDs to download. Default: caller + agent voices ("
                             f"{DEFAULT_CALLER_VOICE}, {DEFAULT_AGENT_VOICE}).")
    parser.add_argument("--list", action="store_true", help="Show available voices and exit.")
    parser.add_argument("--dest", type=Path, default=PIPER_MODELS_DIR,
                        help=f"Destination dir (default: {PIPER_MODELS_DIR})")
    args = parser.parse_args()

    if args.list:
        list_voices()
        return 0

    requested = args.voices or [DEFAULT_CALLER_VOICE, DEFAULT_AGENT_VOICE]
    unknown = [v for v in requested if v not in VOICES]
    if unknown:
        print(f"Unknown voice(s): {unknown}", file=sys.stderr)
        print(f"Run with --list to see available voices.", file=sys.stderr)
        return 2

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(requested)} voice(s) to {args.dest}")

    for vid in requested:
        ok = download_voice(vid, args.dest)
        if not ok:
            return 1

    print(f"\n✅ Done. Voices in {args.dest}.")
    print(f"   Activate Piper TTS:  export PRIVACY_GAME_TTS=piper")
    print(f"   (or leave it on 'auto' — defaults to Piper when available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
