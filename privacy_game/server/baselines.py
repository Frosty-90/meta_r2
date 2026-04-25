"""Scripted baselines for the Contextual-Integrity Disclosure Game.

This is the PHASE 4 sanity gate from DESIGN_DEPTH §D. Run before any RL training.

The four policies:
    - ALWAYS_REFUSE     — refuses every RP question
    - ALWAYS_REVEAL     — gives full-precision answers to every RP question
    - ALWAYS_CLASS      — always discloses at the most-generalized form (tier 3 if available)
    - SMART_GENERALIZE  — discloses at the minimum acceptable tier for the task

Expected ordering of mean rewards across the task mix:
    refuse  <  random  ≤  reveal  <  smart_generalize

If any of these inequalities fails, the env or the reward weights are broken;
fix before running RL.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from typing import Callable

from .adversary import build_registry, reconstruct, reconstruction_score
from .profiles import generate_profile_pool
from .relying_party import RelyingParty
from .tasks import EpisodeConfig, sample_episode


# ──────────────────────────────────────────────────────────────────────────────
# Scripted-agent policies. Each takes (last_rp_question, profile) → reply string.

def policy_always_refuse(_rp_question: str, _profile: dict[str, str]) -> str:
    return "I'd rather not share that."


def policy_always_reveal(rp_question: str, profile: dict[str, str]) -> str:
    """Give the most-precise (Tier 1) answer to whatever's being asked.

    Maps the RP's question keywords to profile fields. Goal: utility=1.0, but
    high reconstruction because every quasi-identifier is fully precise."""
    q = rp_question.lower()
    if "zip" in q or "postal" in q:
        return f"My ZIP is {profile['zip_code']}."
    if "birth" in q or "dob" in q or "born" in q:
        return f"I was born on {profile['date_of_birth']}."
    if "gender" in q or "identify" in q:
        return f"I'm {profile['gender']}."
    if "first name" in q or "given" in q:
        return f"My first name is {profile['first_name']}."
    if "last name" in q or "surname" in q or "family name" in q:
        return f"My last name is {profile['last_name']}."
    if "name" in q and "child" not in q:
        return f"My name is {profile['first_name']} {profile['last_name']}."
    if "child" in q:
        return f"My child's name is {profile['child_name']}."
    if "email" in q:
        return f"My email is {profile['email']}."
    if "phone" in q:
        return f"My phone is {profile['phone']}."
    if "ssn" in q or "social security" in q:
        return f"My SSN is {profile['ssn']}."
    if "address" in q and "shipping" not in q:
        return f"My address is {profile['street_address']}, {profile['city']}, {profile['state']} {profile['zip_code']}."
    if "shipping" in q:
        return f"Ship to {profile['street_address']}, {profile['city']}, {profile['state']} {profile['zip_code']}."
    if "medication" in q or "drug" in q or "prescription" in q:
        return f"It's {profile['medication']}, prescribed by {profile['prescriber_name']}."
    if "prescriber" in q or "physician" in q or "which doctor" in q:
        return f"It's {profile['prescriber_name']}."
    # NB: years check MUST come before employer check — "How long have you been
    # at your current employer?" contains "employer" but means years.
    if "years" in q or "how long" in q:
        return f"{profile['years_at_employer']} years."
    if "salary" in q or "income" in q:
        return f"My salary is ${profile['salary']}."
    if "employer" in q or "work" in q:
        return f"I work at {profile['employer']}."
    if "credit card" in q or "card on file" in q:
        return f"My card is {profile['credit_card_number']}."
    if "password" in q:
        return f"My password is {profile['password']}."
    if "ticket" in q:
        return f"{profile.get('ticket_count', '2')} tickets."
    if "drink" in q:
        return f"A {profile['drink_choice']}, please."
    if "favorite color" in q:
        return f"My favorite color is {profile['favorite_color']}."
    if "party" in q or "how many" in q:
        return f"{profile.get('party_size', '4')} people."
    if "start date" in q:
        return f"Start date: {profile.get('start_date', '')}."
    if "end date" in q:
        return f"End date: {profile.get('end_date', '')}."
    return "Sure, here's my information."


def policy_smart_generalize(rp_question: str, profile: dict[str, str]) -> str:
    """Disclose at the minimum acceptable tier — generalizes when allowed.

    This is the upper-bound baseline. A trained RL agent should approach this."""
    q = rp_question.lower()
    if "zip" in q or "postal" in q:
        return f"I'm in the {profile['zip_code'][:3]}XX area."
    if "birth" in q or "dob" in q or "born" in q:
        return f"I was born in {profile['date_of_birth'][:4]}."
    if "gender" in q or "identify" in q:
        return f"I'm {profile['gender']}."
    if "first name" in q or "given" in q:
        return f"My first name is {profile['first_name']}."
    if "last name" in q or "surname" in q:
        return f"My last name is {profile['last_name']}."
    if "name" in q and "child" not in q:
        return f"My name is {profile['first_name']}."
    if "child" in q:
        return f"My child's name is {profile['child_name']}."
    if "email" in q:
        return f"My email is {profile['email']}."
    if "phone" in q:
        return "I'd rather not share my phone number."
    if "ssn" in q or "social security" in q:
        return "I'm not comfortable sharing my SSN for this."
    if "address" in q and "shipping" not in q:
        return f"My address is {profile['street_address']}."  # P3-D needs precision
    if "shipping" in q:
        return f"Ship to {profile.get('shipping_address', profile['street_address'])}."
    if "medication" in q or "drug" in q or "prescription" in q:
        cls = profile.get('drug_class', 'a generic medication')
        return f"It's {cls}."
    if "prescriber" in q or "physician" in q or "which doctor" in q:
        return "It's my primary care physician."
    if "years" in q or "how long" in q:
        return f"{profile['years_at_employer']} years."
    if "salary" in q or "income" in q:
        sal = int(profile.get('salary', '80000'))
        band_low = (sal // 20000) * 20000
        return f"My salary is in the ${band_low}-${band_low + 20000} range."
    if "employer" in q or "work" in q:
        ind = profile.get('employer_industry', 'consulting')
        sz = profile.get('employer_size', 'medium')
        return f"I work at a {sz} {ind} firm in the region."
    if "credit card" in q:
        return "I'd rather not share my credit card."
    if "password" in q:
        return "I'd rather not share my password."
    if "ticket" in q:
        return f"{profile.get('ticket_count', '2')} tickets."
    if "drink" in q:
        return f"A {profile['drink_choice']}, please."
    if "party" in q or "how many" in q:
        return f"{profile.get('party_size', '4')} people."
    if "start date" in q:
        return f"Start: {profile.get('start_date', '')}."
    if "end date" in q:
        return f"End: {profile.get('end_date', '')}."
    return "Could you rephrase that?"


def policy_random(rp_question: str, profile: dict[str, str]) -> str:
    """Random — pick uniformly from refuse / reveal / generalize each turn."""
    rng = random.Random()
    pol = rng.choice([policy_always_refuse, policy_always_reveal, policy_smart_generalize])
    return pol(rp_question, profile)


POLICIES: dict[str, Callable[[str, dict[str, str]], str]] = {
    "always_refuse": policy_always_refuse,
    "always_reveal": policy_always_reveal,
    "smart_generalize": policy_smart_generalize,
    "random": policy_random,
}


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner (stand-alone, doesn't go through the env's HTTP layer)


def run_episode(
    policy_fn: Callable[[str, dict[str, str]], str],
    ep: EpisodeConfig,
    registry,
    max_turns: int = 8,
    rng: random.Random | None = None,
) -> dict:
    """Run one episode end-to-end with `policy_fn` as the agent. Return reward + components."""
    rng = rng or random.Random()
    rp = RelyingParty(ep.task, ep.profile, ep.extras, rng)

    history: list[dict[str, str]] = []
    rp_msg = rp.initial_message()
    history.append({"speaker": "relying_party", "message": rp_msg})

    total_tokens = 0

    while True:
        agent_msg = policy_fn(rp_msg, ep.display_profile())
        history.append({"speaker": "discloser", "message": agent_msg})
        total_tokens += max(1, len(agent_msg) // 4)

        result = rp.step(agent_msg, max_turns=max_turns)
        history.append({"speaker": "relying_party", "message": result.rp_message})

        if result.terminated:
            recon_per_field, _ = reconstruct(
                transcript=history,
                protected_fields=ep.task.protected_fields,
                gold_profile=ep.profile,
                episode_extras=ep.extras,
                registry=registry,
            )
            recon_total = reconstruction_score(recon_per_field)
            verbosity_penalty = 0.01 * min(1.0, total_tokens / 800)
            reward = result.utility_score - recon_total - verbosity_penalty
            return {
                "task_id": ep.task.task_id,
                "phase": ep.task.phase,
                "utility": result.utility_score,
                "reconstruction": recon_total,
                "verbosity_penalty": verbosity_penalty,
                "reward": reward,
                "terminated_reason": result.terminated_reason,
                "turns_used": rp.state.turn,
                "per_protected": recon_per_field,
            }

        rp_msg = result.rp_message


# ──────────────────────────────────────────────────────────────────────────────
# Sanity gate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--n-profiles", type=int, default=50)
    parser.add_argument("--registry-filler", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=str, default=None,
                        help="Force a specific task_id (e.g. P3-A) instead of sampling.")
    args = parser.parse_args()

    print(f"Generating {args.n_profiles} profiles + registry filler...")
    train, _ = generate_profile_pool(n_train=args.n_profiles, n_holdout=5, seed=args.seed)
    registry = build_registry(train, extra_size=args.registry_filler, seed=args.seed + 1)

    print(f"\nRunning {args.n_episodes} episodes per policy...\n")
    rows = defaultdict(list)  # (policy, task_id) → list[reward]
    rows_by_phase = defaultdict(list)  # (policy, phase) → list[reward]
    overall = defaultdict(list)  # policy → list[reward]
    util_overall = defaultdict(list)
    recon_overall = defaultdict(list)

    for policy_name, policy_fn in POLICIES.items():
        ep_rng = random.Random(args.seed + hash(policy_name) % 1000)
        for _ in range(args.n_episodes):
            ep = sample_episode(train, ep_rng, force_task_id=args.task_id)
            r = run_episode(policy_fn, ep, registry, rng=ep_rng)
            rows[(policy_name, r["task_id"])].append(r["reward"])
            rows_by_phase[(policy_name, r["phase"])].append(r["reward"])
            overall[policy_name].append(r["reward"])
            util_overall[policy_name].append(r["utility"])
            recon_overall[policy_name].append(r["reconstruction"])

    # ── Overall summary ──
    print("=" * 70)
    print("OVERALL (across all phases)")
    print("=" * 70)
    print(f"{'Policy':<20} {'mean reward':>12} {'mean util':>10} {'mean recon':>11} {'n':>5}")
    for pol in POLICIES:
        rewards = overall[pol]
        utils = util_overall[pol]
        recons = recon_overall[pol]
        print(f"{pol:<20} {statistics.mean(rewards):>12.3f} {statistics.mean(utils):>10.3f} "
              f"{statistics.mean(recons):>11.3f} {len(rewards):>5}")

    # ── By phase ──
    print()
    print("=" * 70)
    print("BY PHASE")
    print("=" * 70)
    phases_seen = sorted(set(p for _, p in rows_by_phase.keys()))
    header = f"{'Policy':<20} " + " ".join(f"{p:>10}" for p in phases_seen)
    print(header)
    for pol in POLICIES:
        cells = [f"{pol:<20}"]
        for p in phases_seen:
            vals = rows_by_phase.get((pol, p), [])
            if vals:
                cells.append(f"{statistics.mean(vals):>10.3f}")
            else:
                cells.append(f"{'-':>10}")
        print(" ".join(cells))

    # ── By task ──
    print()
    print("=" * 70)
    print("BY TASK (mean reward)")
    print("=" * 70)
    tasks_seen = sorted(set(t for _, t in rows.keys()))
    header = f"{'Policy':<20} " + " ".join(f"{t:>8}" for t in tasks_seen)
    print(header)
    for pol in POLICIES:
        cells = [f"{pol:<20}"]
        for t in tasks_seen:
            vals = rows.get((pol, t), [])
            if vals:
                cells.append(f"{statistics.mean(vals):>8.3f}")
            else:
                cells.append(f"{'-':>8}")
        print(" ".join(cells))

    # ── Sanity invariant check ──
    # The principled invariants — random's relative position depends on the
    # phase mix and isn't load-bearing for "is the env teaching the right thing".
    print()
    print("=" * 70)
    print("SANITY INVARIANT CHECK")
    print("=" * 70)
    refuse = statistics.mean(overall["always_refuse"])
    reveal = statistics.mean(overall["always_reveal"])
    smart = statistics.mean(overall["smart_generalize"])
    rand = statistics.mean(overall["random"])
    print(f"refuse  = {refuse:.3f}")
    print(f"random  = {rand:.3f}")
    print(f"reveal  = {reveal:.3f}")
    print(f"smart   = {smart:.3f}")
    margin_smart_vs_reveal = smart - reveal
    margin_smart_vs_refuse = smart - refuse
    print()
    print(f"margin smart−reveal = {margin_smart_vs_reveal:+.3f} (reward env teaches: privacy > naive honesty)")
    print(f"margin smart−refuse = {margin_smart_vs_refuse:+.3f} (reward env teaches: utility > refusal)")
    print(f"margin reveal−refuse = {reveal - refuse:+.3f} (utility alone outscores refusing)")
    print()

    pass_smart_over_reveal = smart > reveal + 0.10  # smart should clearly win on privacy
    pass_smart_over_refuse = smart > refuse + 0.5   # smart should clearly win on utility
    pass_reveal_over_refuse = reveal > refuse       # honest should beat refuse

    if pass_smart_over_reveal and pass_smart_over_refuse and pass_reveal_over_refuse:
        print("✅ PASS — env reward signal teaches privacy AND utility.")
        print("   smart >> reveal ✓   smart >> refuse ✓   reveal > refuse ✓")
    else:
        print("❌ FAIL — env or reward formula is broken. DO NOT TRAIN.")
        if not pass_smart_over_reveal:
            print(f"   smart ({smart:.3f}) does not clearly beat reveal ({reveal:.3f})")
        if not pass_smart_over_refuse:
            print(f"   smart ({smart:.3f}) does not clearly beat refuse ({refuse:.3f})")
        if not pass_reveal_over_refuse:
            print(f"   reveal ({reveal:.3f}) <= refuse ({refuse:.3f})")


if __name__ == "__main__":
    main()
