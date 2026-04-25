"""Cross-modality evaluation: does the text-level adversarial signal hold up
when the adversary reconstructs from Whisper-transcribed audio?

This is the key research question for the voice extension.

Procedure, for N forced-task episodes per (policy, task) pair:
    1. Run the episode text-first (same as baselines.run_episode) — record
       oracle text transcript + gold reward components.
    2. Render every utterance as audio via macOS `say` with distinct voices.
    3. Concatenate per-turn WAVs into one full conversation WAV.
    4. Transcribe the full WAV back via Whisper (end-to-end ASR).
    5. Apply the SAME adversary to the WHISPER transcript. Record reconstruction.
    6. Compare (oracle-text recon) vs (voice recon).

Output:
    per-condition reconstruction rate, reward, and Δ(voice − oracle) per task.
    Per-episode CSV for reproducibility.
    Side-by-side rendered transcripts for qualitative inspection.

Interpretation:
    - If voice_recon ≈ oracle_recon: the policy generalizes to voice cleanly.
    - If voice_recon < oracle_recon: Whisper noise accidentally defends the
      agent; the modality ADDS privacy. Not a real agent skill — document
      honestly.
    - If voice_recon > oracle_recon: Whisper mis-transcribes in ways that
      INVITE additional leaks (e.g. phonetic name collisions). Rare but
      possible. If observed, flag as voice-specific vulnerability.
"""

from __future__ import annotations

import csv
import random
import shutil
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..server.adversary import (
    build_registry,
    reconstruct,
    reconstruction_score,
)
from ..server.baselines import (
    policy_always_refuse,
    policy_always_reveal,
    policy_smart_generalize,
)
from ..server.profiles import (
    LEAKY_EMPLOYERS,
    STRONG_DIAG_MEDS,
    generate_profile_pool,
)
from ..server.tasks import ALL_TASKS_BY_ID, EpisodeConfig, sample_episode
from .asr import default_asr
from .voice_shell import render_episode_as_voice


@dataclass
class CrossModalRow:
    task_id: str
    policy: str
    profile_id: str
    oracle_utility: float
    oracle_recon: float
    oracle_reward: float
    voice_recon: float
    voice_reward: float
    delta_recon: float     # voice_recon - oracle_recon
    delta_reward: float    # voice_reward - oracle_reward


def _pick_profile_for_task(train: list[dict], task_id: str) -> dict:
    """Pick a profile that makes the task's leak mechanism plausibly fire."""
    if task_id == "P3-B":
        strong = {d for d, _, _ in STRONG_DIAG_MEDS}
        for p in train:
            if p.get("medication") in strong:
                return p
    if task_id == "P3-C":
        leaky = {emp for emp, _, _ in LEAKY_EMPLOYERS}
        for p in train:
            if p.get("employer") in leaky:
                return p
    return train[0]


def _voice_adversary_scan(
    combined_wav: Path,
    protected_fields: list[str],
    gold_profile: dict,
    episode_extras: dict,
    registry,
) -> tuple[float, dict, dict]:
    """Transcribe the full conversation WAV and run the adversary on the
    reconstructed text."""
    asr = default_asr()
    transcript = asr.transcribe(combined_wav).text
    # We don't know who said what after re-transcription, so build a single-
    # speaker pseudo-transcript that the adversary parses holistically.
    pseudo_history = [{"speaker": "combined", "message": transcript}]
    scores, recovered = reconstruct(
        transcript=pseudo_history,
        protected_fields=protected_fields,
        gold_profile=gold_profile,
        episode_extras=episode_extras,
        registry=registry,
    )
    return reconstruction_score(scores), scores, recovered


def main(
    n_profiles: int = 10,
    tasks: tuple[str, ...] = ("P3-A", "P3-B", "P3-C", "P3-D"),
    policies: tuple[tuple[str, object], ...] = (
        ("reveal", policy_always_reveal),
        ("smart", policy_smart_generalize),
        ("refuse", policy_always_refuse),
    ),
    seed: int = 42,
    out_dir: str = "/tmp/privacy_game_voice_eval",
):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, _ = generate_profile_pool(n_train=200, n_holdout=20, seed=seed)
    registry = build_registry(train, extra_size=2000, seed=seed + 1)

    # Preload Whisper once
    _ = default_asr()

    rows: list[CrossModalRow] = []
    rng = random.Random(seed)

    total_runs = n_profiles * len(tasks) * len(policies)
    run_i = 0

    for task_id in tasks:
        task = ALL_TASKS_BY_ID[task_id]
        # For each task, pick up to n_profiles that make the task's leak fire
        if task_id == "P3-B":
            strong = {d for d, _, _ in STRONG_DIAG_MEDS}
            pool = [p for p in train if p.get("medication") in strong]
        elif task_id == "P3-C":
            leaky = {emp for emp, _, _ in LEAKY_EMPLOYERS}
            pool = [p for p in train if p.get("employer") in leaky]
        else:
            pool = train
        # Deterministic selection
        selected = pool[:n_profiles] if len(pool) >= n_profiles else pool

        for profile in selected:
            for policy_name, policy_fn in policies:
                run_i += 1
                print(f"[{run_i}/{total_runs}] task={task_id} policy={policy_name} profile={profile['profile_id']}")

                seed_ep = sample_episode(train, rng, force_task_id=task_id)
                ep = EpisodeConfig(
                    task=task,
                    profile=profile,
                    extras=seed_ep.extras,
                    home_city=profile.get("city"),
                )

                # Augment reveal on P3-D with a destination mention (so over-share has something to catch)
                effective_policy = policy_fn
                if task_id == "P3-D" and policy_name == "reveal":
                    _orig = policy_fn
                    def effective_policy(q, p, _orig=_orig):
                        reply = _orig(q, p)
                        if any(k in q.lower() for k in ("date", "start", "end")):
                            reply = reply + " Excited about our trip to Paris!"
                        return reply

                label = f"{task_id}_{policy_name}_{profile['profile_id']}"
                scenario_dir = out_dir / label
                scenario_dir.mkdir(parents=True, exist_ok=True)

                # Render voice episode (also scores on oracle text)
                result = render_episode_as_voice(
                    policy_fn=effective_policy,
                    ep=ep,
                    out_dir=scenario_dir,
                    registry=registry,
                    label=label,
                )

                # Voice-side adversary: transcribe the combined WAV and rescore
                voice_recon, voice_scores, voice_recovered = _voice_adversary_scan(
                    combined_wav=result.combined_wav,
                    protected_fields=task.protected_fields,
                    gold_profile=profile,
                    episode_extras=ep.extras,
                    registry=registry,
                )
                voice_reward = result.utility - voice_recon - result.verbosity_penalty

                rows.append(CrossModalRow(
                    task_id=task_id,
                    policy=policy_name,
                    profile_id=profile["profile_id"],
                    oracle_utility=result.utility,
                    oracle_recon=result.reconstruction,
                    oracle_reward=result.reward,
                    voice_recon=voice_recon,
                    voice_reward=voice_reward,
                    delta_recon=voice_recon - result.reconstruction,
                    delta_reward=voice_reward - result.reward,
                ))

                print(f"    oracle: recon={result.reconstruction:.2f} reward={result.reward:+.3f}")
                print(f"    voice:  recon={voice_recon:.2f} reward={voice_reward:+.3f}")
                print(f"    Δrecon={rows[-1].delta_recon:+.2f} Δreward={rows[-1].delta_reward:+.3f}")

    # ── CSV ──
    csv_path = out_dir / "cross_modal_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # ── Aggregated summary ──
    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.task_id, r.policy)
        agg[key]["oracle_recon"].append(r.oracle_recon)
        agg[key]["voice_recon"].append(r.voice_recon)
        agg[key]["oracle_reward"].append(r.oracle_reward)
        agg[key]["voice_reward"].append(r.voice_reward)

    summary_lines = ["# Cross-Modality Audio Evaluation\n"]
    summary_lines.append(f"N={n_profiles} profiles × {len(tasks)} tasks × {len(policies)} policies = {len(rows)} episodes\n\n")
    summary_lines.append("## Reconstruction rate: oracle text vs Whisper-transcribed voice\n\n")
    summary_lines.append("| Task | Policy | Oracle recon | Voice recon | Δrecon | Oracle reward | Voice reward | Δreward |\n")
    summary_lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for (task_id, policy_name), vals in sorted(agg.items()):
        summary_lines.append(
            f"| {task_id} | {policy_name} | "
            f"{statistics.mean(vals['oracle_recon']):.3f} | {statistics.mean(vals['voice_recon']):.3f} | "
            f"{statistics.mean(vals['voice_recon']) - statistics.mean(vals['oracle_recon']):+.3f} | "
            f"{statistics.mean(vals['oracle_reward']):+.3f} | {statistics.mean(vals['voice_reward']):+.3f} | "
            f"{statistics.mean(vals['voice_reward']) - statistics.mean(vals['oracle_reward']):+.3f} |\n"
        )
    summary_lines.append("\n## Interpretation\n\n")
    summary_lines.append(
        "**If Δrecon ≈ 0**: the policy's privacy/utility behavior generalizes across modalities — "
        "the skill is about information content, not text syntax. This is the strongest outcome.\n\n"
    )
    summary_lines.append(
        "**If Δrecon < 0** (voice leaks less than oracle): Whisper ASR errors accidentally scrub some "
        "leaked values. The agent didn't earn this privacy — the modality did. Document honestly.\n\n"
    )
    summary_lines.append(
        "**If Δrecon > 0** (voice leaks more): Whisper mis-transcribes in ways that invite new leaks "
        "(phonetic name collisions, digit confusions). A voice-specific vulnerability.\n"
    )

    summary_path = out_dir / "summary.md"
    summary_path.write_text("".join(summary_lines))

    print(f"\n=== DONE ===")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"\nPreview (first few lines of summary):")
    for line in summary_lines[:20]:
        print("  " + line.rstrip())


if __name__ == "__main__":
    main(n_profiles=3)  # start with 3 per task × 3 policies = 36 episodes for smoke
