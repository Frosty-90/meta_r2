"""End-to-end voice layer over a Disclosure Game episode.

Given a scripted or trained agent policy, runs the episode text-first (same
semantics as baselines.run_episode) and ALSO renders every utterance as audio
using distinct voices for the caller (Relying Party) and the agent (Discloser).
Saves per-turn WAVs + one concatenated `full_conversation.wav` for demo video.

Research framing: reward is always computed on TEXT (via the same adversary
that sees the text transcript). The voice layer is a deployment surface, not a
new training substrate. If a text-trained policy still behaves well when
rendered through voice, that's evidence the skill is modality-agnostic
information control — stronger than "the model learned a text game."
"""

from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..server.adversary import build_registry, reconstruct, reconstruction_score
from ..server.profiles import generate_profile_pool
from ..server.relying_party import RelyingParty
from ..server.tasks import ALL_TASKS_BY_ID, EpisodeConfig, sample_episode
from .tts import AGENT_VOICE, CALLER_VOICE, Voice, synthesize_to_file


@dataclass
class VoiceTurn:
    """One utterance in the voice rendering."""

    speaker: str           # "caller" (relying party) or "agent" (discloser)
    text: str
    wav_path: Path
    duration_sec: float = 0.0  # filled in after synthesis


@dataclass
class VoiceEpisodeResult:
    """Output of a full voice episode rendering."""

    task_id: str
    phase: str
    profile: dict[str, str]
    turns: list[VoiceTurn] = field(default_factory=list)

    # Text-level scoring — SAME as baselines.run_episode
    utility: float = 0.0
    reconstruction: float = 0.0
    verbosity_penalty: float = 0.0
    reward: float = 0.0
    terminated_reason: Optional[str] = None
    per_protected: dict[str, float] = field(default_factory=dict)
    per_protected_recovered: dict[str, Optional[str]] = field(default_factory=dict)

    # Voice-level assets
    combined_wav: Optional[Path] = None
    transcript_txt: Optional[Path] = None


def _concat_wavs_ffmpeg(wav_paths: list[Path], out_path: Path, silence_ms: int = 400) -> Path:
    """Concatenate WAVs with a short silence gap between turns, via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not installed — required for voice concat")

    # Build a silence file of the right duration
    silence = out_path.parent / "_silence.wav"
    if not silence.exists():
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", f"{silence_ms / 1000.0:.3f}",
            "-c:a", "pcm_s16le",
            str(silence),
        ], check=True)

    # Write a concat list file (ffmpeg concat demuxer)
    listfile = out_path.parent / "_concat.txt"
    lines = []
    for i, p in enumerate(wav_paths):
        lines.append(f"file '{p.as_posix()}'")
        if i < len(wav_paths) - 1:
            lines.append(f"file '{silence.as_posix()}'")
    listfile.write_text("\n".join(lines))

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(listfile),
        "-c:a", "pcm_s16le", "-ar", "16000",
        str(out_path),
    ], check=True)

    # Cleanup
    listfile.unlink(missing_ok=True)
    return out_path


def _wav_duration(path: Path) -> float:
    """Return WAV duration in seconds using soundfile."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return 0.0


def render_episode_as_voice(
    policy_fn: Callable[[str, dict[str, str]], str],
    ep: EpisodeConfig,
    out_dir: Path,
    registry,
    max_turns: int = 8,
    caller_voice: Voice = CALLER_VOICE,
    agent_voice: Voice = AGENT_VOICE,
    rng: Optional[random.Random] = None,
    label: str = "episode",
) -> VoiceEpisodeResult:
    """Run an episode and render every utterance as audio. Returns full result.

    Args:
        policy_fn: (rp_message, profile_dict) -> agent reply text
        ep: forced episode config
        out_dir: directory to drop per-turn WAVs, combined WAV, transcript .txt
        label: prefix for filenames (e.g. "base_model_P3-A" vs "trained_P3-A")
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random(0)

    result = VoiceEpisodeResult(
        task_id=ep.task.task_id,
        phase=ep.task.phase,
        profile=dict(ep.profile),
    )

    rp = RelyingParty(ep.task, ep.profile, ep.extras, rng)
    rp_msg = rp.initial_message()
    history: list[dict[str, str]] = [{"speaker": "relying_party", "message": rp_msg}]

    # Render initial RP turn
    wav0 = out_dir / f"{label}_000_caller.wav"
    synthesize_to_file(rp_msg, wav0, voice=caller_voice)
    result.turns.append(VoiceTurn(
        speaker="caller", text=rp_msg, wav_path=wav0, duration_sec=_wav_duration(wav0),
    ))

    total_tokens = 0
    turn_idx = 0
    while True:
        turn_idx += 1

        # Agent turn
        agent_msg = policy_fn(rp_msg, ep.display_profile())
        history.append({"speaker": "discloser", "message": agent_msg})
        total_tokens += max(1, len(agent_msg) // 4)
        wav_agent = out_dir / f"{label}_{turn_idx:03d}_agent.wav"
        synthesize_to_file(agent_msg, wav_agent, voice=agent_voice)
        result.turns.append(VoiceTurn(
            speaker="agent", text=agent_msg, wav_path=wav_agent,
            duration_sec=_wav_duration(wav_agent),
        ))

        # Advance RP
        rp_result = rp.step(agent_msg, max_turns=max_turns)
        history.append({"speaker": "relying_party", "message": rp_result.rp_message})
        wav_rp = out_dir / f"{label}_{turn_idx:03d}_caller.wav"
        synthesize_to_file(rp_result.rp_message, wav_rp, voice=caller_voice)
        result.turns.append(VoiceTurn(
            speaker="caller", text=rp_result.rp_message, wav_path=wav_rp,
            duration_sec=_wav_duration(wav_rp),
        ))

        if rp_result.terminated:
            # Score the episode on TEXT (same as baselines)
            recon_scores, recovered = reconstruct(
                transcript=history,
                protected_fields=ep.task.protected_fields,
                gold_profile=ep.profile,
                episode_extras=ep.extras,
                registry=registry,
            )
            recon_total = reconstruction_score(recon_scores)
            verbosity_penalty = 0.01 * min(1.0, total_tokens / 800)
            reward = rp_result.utility_score - recon_total - verbosity_penalty

            result.utility = rp_result.utility_score
            result.reconstruction = recon_total
            result.verbosity_penalty = verbosity_penalty
            result.reward = reward
            result.terminated_reason = rp_result.terminated_reason
            result.per_protected = recon_scores
            result.per_protected_recovered = recovered
            break

        rp_msg = rp_result.rp_message

    # Concat turn WAVs into one full conversation WAV
    combined = out_dir / f"{label}_full_conversation.wav"
    _concat_wavs_ffmpeg([t.wav_path for t in result.turns], combined)
    result.combined_wav = combined

    # Save full transcript as .txt for demo captions
    transcript_txt = out_dir / f"{label}_transcript.txt"
    lines = [f"=== {label} | task={result.task_id} ({result.phase}) ==="]
    lines.append(f"Profile: {ep.profile.get('full_name', '?')} | zip={ep.profile.get('zip_code')} "
                 f"dob={ep.profile.get('date_of_birth')} gender={ep.profile.get('gender')}")
    lines.append(f"Required: {[f for f, _ in ep.task.required_with_tiers]}")
    lines.append(f"Protected: {ep.task.protected_fields}")
    lines.append("")
    for t in result.turns:
        speaker_label = "CALLER" if t.speaker == "caller" else "AGENT "
        lines.append(f"{speaker_label}: {t.text}")
    lines.append("")
    lines.append(f"--- RESULT ---")
    lines.append(f"reason:         {result.terminated_reason}")
    lines.append(f"utility:        {result.utility}")
    lines.append(f"reconstruction: {result.reconstruction:.3f}")
    lines.append(f"reward:         {result.reward:.3f}")
    lines.append(f"recovered:      {result.per_protected_recovered}")
    transcript_txt.write_text("\n".join(lines))
    result.transcript_txt = transcript_txt

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: a single-call demo runner


def demo_episode(
    policy_name: str,
    policy_fn: Callable,
    task_id: str,
    out_dir: Path,
    profile_predicate: Optional[Callable[[dict], bool]] = None,
    seed: int = 42,
) -> VoiceEpisodeResult:
    """Render one forced-task episode and return the result. Useful for demos."""
    train, _ = generate_profile_pool(n_train=50, n_holdout=5, seed=seed)
    registry = build_registry(train, extra_size=1000, seed=seed + 1)
    if profile_predicate:
        profile = next(p for p in train if profile_predicate(p))
    else:
        profile = train[0]

    rng = random.Random(seed)
    task = ALL_TASKS_BY_ID[task_id]
    ep = EpisodeConfig(
        task=task,
        profile=profile,
        extras={},
        home_city=profile.get("city"),
    )
    # Re-populate extras if the task needs any (dates, ticket_count, etc.)
    ep = sample_episode(train, rng, force_task_id=task_id)
    ep = EpisodeConfig(task=task, profile=profile, extras=ep.extras, home_city=profile.get("city"))

    return render_episode_as_voice(
        policy_fn=policy_fn,
        ep=ep,
        out_dir=out_dir,
        registry=registry,
        label=f"{policy_name}_{task_id}",
    )


if __name__ == "__main__":
    # Quick smoke test — render P3-A with the smart policy into WAVs
    from ..server.baselines import policy_smart_generalize

    out_dir = Path("/tmp/privacy_game_voice_demo")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering smart_generalize episode of P3-A as voice...")
    result = demo_episode("smart", policy_smart_generalize, "P3-A", out_dir)

    print(f"\n=== RESULT ===")
    print(f"task = {result.task_id} ({result.phase})")
    print(f"profile = {result.profile['full_name']} | zip={result.profile['zip_code']} dob={result.profile['date_of_birth']}")
    print(f"utility = {result.utility}  reconstruction = {result.reconstruction:.3f}  reward = {result.reward:.3f}")
    print(f"per_protected_recovered = {result.per_protected_recovered}")
    print(f"\n  combined WAV: {result.combined_wav}  ({_wav_duration(result.combined_wav):.1f}s)")
    print(f"  transcript:   {result.transcript_txt}")
    print(f"  turns:        {len(result.turns)}")
