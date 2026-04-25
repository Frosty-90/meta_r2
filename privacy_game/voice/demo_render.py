"""Pre-render the hackathon demo scenarios as voice WAVs.

Produces 8 demonstration audio files comparing `always_reveal` (naive base-
model-style disclosure) against `smart_generalize` (target behavior of a
trained RL agent) across all four P3 research tasks.

Each scenario picks a profile *designed* to exhibit the leak — e.g. for P3-A
the chosen profile's (zip, dob, gender) triple is unique in the registry; for
P3-B the profile's medication is strongly diagnostic; for P3-C the profile
has a leaky employer.

Output:
    /tmp/privacy_game_demo/<policy>_<task>_full_conversation.wav
    /tmp/privacy_game_demo/<policy>_<task>_transcript.txt
    /tmp/privacy_game_demo/summary.md               (human-readable summary + comparison table)

Usage:
    python -m privacy_game.voice.demo_render
    # then for demo video:
    afplay /tmp/privacy_game_demo/reveal_P3-A_full_conversation.wav   # leaky base
    afplay /tmp/privacy_game_demo/smart_P3-A_full_conversation.wav    # defended trained
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..server.adversary import build_registry
from ..server.baselines import policy_always_reveal, policy_smart_generalize
from ..server.profiles import LEAKY_EMPLOYERS, STRONG_DIAG_MEDS, generate_profile_pool
from ..server.tasks import ALL_TASKS_BY_ID, EpisodeConfig, sample_episode
from .voice_shell import render_episode_as_voice


# ──────────────────────────────────────────────────────────────────────────────
# Scenario selection — pick profiles that make each task's leak *actually fire*

def pick_p3a_profile(profiles):
    """P3-A: any profile works (Sweeney fires for unique triples)."""
    return profiles[0]


def pick_p3b_profile(profiles):
    """P3-B: need a strongly-diagnostic medication."""
    strong_drugs = {d for d, _, _ in STRONG_DIAG_MEDS}
    for p in profiles:
        if p.get("medication") in strong_drugs:
            return p
    return profiles[0]


def pick_p3c_profile(profiles):
    """P3-C: need a LEAKY employer (for the contrast to be real)."""
    leaky = {emp for emp, _, _ in LEAKY_EMPLOYERS}
    for p in profiles:
        if p.get("employer") in leaky:
            return p
    return profiles[0]


def pick_p3d_profile(profiles):
    """P3-D: need a NON-home city that the leak-policy will mention — we force
    the `reveal` policy to mention a vacation destination in the narration."""
    # Any profile works; leak fires because always_reveal volunteers extra
    return profiles[0]


SCENARIOS = [
    ("P3-A", "Insurance underwriting (k-anonymity / Sweeney triangulation)", pick_p3a_profile),
    ("P3-B", "Pharmacy verification (drug → diagnosis inference)", pick_p3b_profile),
    ("P3-C", "Apartment rental (employer → religion / health / politics)", pick_p3c_profile),
    ("P3-D", "Vacation mail hold (omission discipline / over-share scan)", pick_p3d_profile),
]


# ──────────────────────────────────────────────────────────────────────────────
# Main


def main():
    out_dir = Path("/tmp/privacy_game_demo")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, _ = generate_profile_pool(n_train=150, n_holdout=20, seed=42)
    registry = build_registry(train, extra_size=2000, seed=43)

    policies = [
        ("reveal", policy_always_reveal),
        ("smart", policy_smart_generalize),
    ]

    rows = []   # for summary.md

    for task_id, description, profile_picker in SCENARIOS:
        profile = profile_picker(train)
        task = ALL_TASKS_BY_ID[task_id]

        for policy_name, policy_fn in policies:
            print(f"\n=== Rendering {policy_name} on {task_id} ===")
            print(f"  profile: {profile['full_name']}")
            if task_id == "P3-A":
                print(f"    zip={profile['zip_code']} dob={profile['date_of_birth']} gender={profile['gender']}")
            if task_id == "P3-B":
                print(f"    medication={profile['medication']} → diagnosis={profile['diagnosis']}")
            if task_id == "P3-C":
                print(f"    employer={profile['employer']}")
            if task_id == "P3-D":
                print(f"    home_city={profile['city']}")

            # Build ep — sample to get any required episode_extras (dates, etc)
            rng = random.Random(hash((task_id, policy_name)) & 0xffff)
            seed_ep = sample_episode(train, rng, force_task_id=task_id)
            ep = EpisodeConfig(
                task=task,
                profile=profile,
                extras=seed_ep.extras,
                home_city=profile.get("city"),
            )

            # For P3-D, augment the reveal policy with a destination mention
            # to trigger the over-share leak
            if task_id == "P3-D" and policy_name == "reveal":
                original = policy_fn
                def policy_with_leak(q, p, _orig=original):
                    reply = _orig(q, p)
                    # inject a destination mention into the dates reply
                    if "date" in q.lower() or "start" in q.lower() or "end" in q.lower():
                        reply = reply + " Can't wait for our trip to Paris!"
                    return reply
                active_policy = policy_with_leak
            else:
                active_policy = policy_fn

            result = render_episode_as_voice(
                policy_fn=active_policy,
                ep=ep,
                out_dir=out_dir,
                registry=registry,
                label=f"{policy_name}_{task_id}",
            )

            print(f"  reward={result.reward:.3f}  utility={result.utility}  recon={result.reconstruction:.3f}")
            print(f"  recovered: {result.per_protected_recovered}")
            print(f"  audio:     {result.combined_wav.name}")

            rows.append({
                "task_id": task_id,
                "task_desc": description,
                "policy": policy_name,
                "profile_name": profile["full_name"],
                "profile_key_fields": _profile_key_fields(profile, task_id),
                "reward": result.reward,
                "utility": result.utility,
                "reconstruction": result.reconstruction,
                "recovered": result.per_protected_recovered,
                "audio_file": result.combined_wav.name,
                "transcript_file": result.transcript_txt.name,
                "duration_sec": sum(t.duration_sec for t in result.turns),
            })

    # ── Write summary.md ──
    summary_path = out_dir / "summary.md"
    lines = ["# Demo Rendering Summary\n"]
    lines.append("Eight voice episodes rendered across four P3 tasks × two policies.\n")
    lines.append("For each task, compare the **reveal** (naive) audio to the **smart** (target) audio — same profile, same caller questions, same adversary scanning the text transcript. The reward axis shows the research claim empirically.\n\n")
    lines.append("| Task | Policy | Profile | Reward | Utility | Recon | Duration | Audio |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---|\n")
    for r in rows:
        lines.append(
            f"| {r['task_id']} | {r['policy']} | {r['profile_name']} | "
            f"{r['reward']:+.3f} | {r['utility']:.1f} | {r['reconstruction']:.3f} | "
            f"{r['duration_sec']:.1f}s | `{r['audio_file']}` |\n"
        )
    lines.append("\n## Side-by-side comparisons\n\n")
    for task_id, task_desc, _ in SCENARIOS:
        reveal_row = next(r for r in rows if r["task_id"] == task_id and r["policy"] == "reveal")
        smart_row = next(r for r in rows if r["task_id"] == task_id and r["policy"] == "smart")
        delta = smart_row["reward"] - reveal_row["reward"]
        lines.append(f"### {task_id} — {task_desc}\n")
        lines.append(f"- **Profile**: {reveal_row['profile_key_fields']}\n")
        lines.append(f"- **Reveal (naive)**: reward={reveal_row['reward']:+.3f}, recon={reveal_row['reconstruction']:.3f}, recovered={reveal_row['recovered']}\n")
        lines.append(f"- **Smart (target)**: reward={smart_row['reward']:+.3f}, recon={smart_row['reconstruction']:.3f}, recovered={smart_row['recovered']}\n")
        lines.append(f"- **Δ reward (smart − reveal)** = {delta:+.3f}\n\n")
    summary_path.write_text("".join(lines))

    print(f"\n\n=== DONE ===")
    print(f"All audio + transcripts in: {out_dir}/")
    print(f"Summary:                    {summary_path}")
    print(f"\nPlay a pair for comparison:")
    print(f"  afplay {out_dir}/reveal_P3-A_full_conversation.wav")
    print(f"  afplay {out_dir}/smart_P3-A_full_conversation.wav")


def _profile_key_fields(profile: dict, task_id: str) -> str:
    if task_id == "P3-A":
        return f"zip={profile['zip_code']} dob={profile['date_of_birth']} gender={profile['gender']}"
    if task_id == "P3-B":
        return f"medication={profile['medication']} (diagnosis={profile['diagnosis']})"
    if task_id == "P3-C":
        return f"employer={profile['employer']} (leaky attr lookup)"
    if task_id == "P3-D":
        return f"home_city={profile['city']}"
    return ""


if __name__ == "__main__":
    main()
