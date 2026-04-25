"""Live interactive demo — FastAPI backend serving both text and voice modes.

Run:
    python -m privacy_game.voice.demo_live
    open http://localhost:8090/

Features:
    - Pick a task (P1-A..P3-D) + reward mode (additive / pareto_it)
    - Text mode: type messages, see RP responses + running disclosure score
    - Voice mode: record via browser mic (MediaRecorder) → Whisper ASR →
      step the env → stream back text + synthesized RP voice reply
    - Live per-protected-field reconstruction view updated every turn
    - Single-page dark-themed UI with chat bubbles

State model: in-memory dict keyed by session_id. Not durable — fine for demos.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..models import DisclosureAction
from ..server.privacy_game_environment import PrivacyGameEnvironment
from ..server.tasks import ALL_TASKS_BY_ID
from .asr import default_asr
from .tts import AGENT_VOICE, CALLER_VOICE, synthesize_to_file


_VALID_REWARD_MODES = {"additive", "pareto_it"}


STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Privacy Game Live Demo")

# In-memory session store. Survives only as long as the server runs.
SESSIONS: dict[str, PrivacyGameEnvironment] = {}


# ── Request / response models ────────────────────────────────────────────────

class StartRequest(BaseModel):
    task_id: Optional[str] = None      # force task; else random sample
    reward_mode: str = "additive"      # "additive" or "pareto_it"


class StartResponse(BaseModel):
    session_id: str
    observation: dict


class StepRequest(BaseModel):
    message: str


class StepResponse(BaseModel):
    observation: dict
    transcribed: Optional[str] = None  # populated by /step-voice


# ── Helpers ──────────────────────────────────────────────────────────────────

def _serialize_obs(obs) -> dict:
    """Pydantic obs → JSON-serializable dict, with metadata flattened."""
    return {
        "task_id": obs.task_id,
        "phase": obs.phase,
        "task_description": obs.task_description,
        "profile": obs.profile,
        "required_fields": obs.required_fields,
        "protected_fields": obs.protected_fields,
        "relying_party_message": obs.relying_party_message,
        "turn_number": obs.turn_number,
        "max_turns": obs.max_turns,
        "history": obs.history,
        "terminated": obs.terminated,
        "terminated_reason": obs.terminated_reason,
        "reward": obs.reward,
        "metadata": obs.metadata or {},
    }


_MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MB cap on uploaded audio (≈ 30 min @ 96kbps)


class AudioDecodeError(Exception):
    """Raised when ffmpeg can't decode the uploaded audio (corrupt / non-audio)."""


def _webm_to_wav16k(webm_bytes: bytes) -> Path:
    """Convert browser-recorded audio (typically webm/opus) to 16kHz mono WAV.

    Raises:
        AudioDecodeError: if ffmpeg fails to decode (corrupt / non-audio bytes).
                          Caller should map this to HTTP 400.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
        f_in.write(webm_bytes)
        in_path = Path(f_in.name)
    out_path = in_path.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_path),
             "-ar", "16000", "-ac", "1", str(out_path)],
            check=True,
            capture_output=True,
            timeout=30,  # cap wall-clock so a malformed input can't hang the server
        )
    except subprocess.TimeoutExpired as e:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        raise AudioDecodeError("audio decode timed out (>30s)") from e
    except subprocess.CalledProcessError as e:
        # ffmpeg exited non-zero — corrupt or non-audio input. Surface as 400, not 500.
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        stderr = (e.stderr or b"").decode(errors="replace")[:200]
        raise AudioDecodeError(f"audio decode failed: {stderr}") from e
    in_path.unlink(missing_ok=True)
    return out_path


# ── API endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {"task_id": t.task_id, "phase": t.phase, "description": t.description}
            for t in ALL_TASKS_BY_ID.values()
        ],
        "reward_modes": ["additive", "pareto_it"],
    }


@app.post("/api/start", response_model=StartResponse)
def start(req: StartRequest) -> StartResponse:
    # Validate inputs at the API boundary so bad client requests get 4xx, not 5xx.
    if req.reward_mode not in _VALID_REWARD_MODES:
        raise HTTPException(
            400,
            f"reward_mode must be one of {sorted(_VALID_REWARD_MODES)}; got {req.reward_mode!r}",
        )
    if req.task_id is not None and req.task_id not in ALL_TASKS_BY_ID:
        raise HTTPException(
            400,
            f"task_id {req.task_id!r} is not a known task. "
            f"Known: {sorted(ALL_TASKS_BY_ID.keys())}",
        )
    env = PrivacyGameEnvironment(
        reward_mode=req.reward_mode,
        force_task_id=req.task_id,
    )
    obs = env.reset()
    sid = str(uuid.uuid4())
    SESSIONS[sid] = env
    return StartResponse(session_id=sid, observation=_serialize_obs(obs))


@app.post("/api/step/{sid}", response_model=StepResponse)
def step_text(sid: str, req: StepRequest) -> StepResponse:
    env = SESSIONS.get(sid)
    if env is None:
        raise HTTPException(404, "session not found")
    obs = env.step(DisclosureAction(message=req.message))
    return StepResponse(observation=_serialize_obs(obs))


@app.post("/api/step-voice/{sid}", response_model=StepResponse)
async def step_voice(sid: str, audio: UploadFile = File(...)) -> StepResponse:
    env = SESSIONS.get(sid)
    if env is None:
        raise HTTPException(404, "session not found")
    blob = await audio.read()
    if not blob:
        raise HTTPException(400, "empty audio upload")
    if len(blob) > _MAX_AUDIO_BYTES:
        raise HTTPException(413, f"audio upload exceeds {_MAX_AUDIO_BYTES // (1024*1024)} MB cap")

    try:
        wav = _webm_to_wav16k(blob)
    except AudioDecodeError as e:
        # Corrupt / non-audio upload — client error, not server failure.
        raise HTTPException(400, str(e))

    try:
        text = default_asr().transcribe(wav).text.strip()
    finally:
        wav.unlink(missing_ok=True)

    if not text:
        text = "(no speech detected)"

    obs = env.step(DisclosureAction(message=text))
    return StepResponse(observation=_serialize_obs(obs), transcribed=text)


@app.get("/api/tts")
def tts(text: str, voice: str = "caller"):
    """Synthesize `text` with the chosen voice. Returns a WAV file."""
    v = CALLER_VOICE if voice == "caller" else AGENT_VOICE
    out = Path(tempfile.mkdtemp()) / "tts.wav"
    synthesize_to_file(text, out, voice=v)
    return FileResponse(out, media_type="audio/wav", filename="tts.wav")


@app.delete("/api/session/{sid}")
def delete_session(sid: str) -> dict:
    SESSIONS.pop(sid, None)
    return {"ok": True}


# ── Startup / CLI ────────────────────────────────────────────────────────────

def main(host: str = "127.0.0.1", port: int = 8090):
    import uvicorn
    # Warm Whisper model at startup so first voice request isn't 10s slow
    print("Warming Whisper base.en model...")
    default_asr()._ensure_loaded()
    print(f"Serving on http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
