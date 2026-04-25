"""Pilot eval — is the env trainable, and is your agent learning?

Two subcommands:

    run     — run N episodes against a named policy, dump trajectories, print stats
    compare — diff two runs (before-vs-after training) and pass/fail it

The staircase
─────────────
    Tier 0  python -m privacy_game.server.baselines             # env-signal gate (existing)
    Tier 1  python -m privacy_game.eval.pilot run ...           # rich per-task stats + trajectory dump
    Tier 2  python -m privacy_game.eval.pilot compare A.jsonl B.jsonl  # before/after delta

Why this exists
───────────────
`baselines.py` already validates the env DISCRIMINATES (smart >> reveal >> refuse).
What it doesn't give you is a per-checkpoint "did my training help?" answer.
This script fills that gap by writing trajectory JSONLs (via the trajectory_logger)
and giving you a `compare` that loads two of them and tells you whether mean reward
moved meaningfully (Welch's t-test on episode rewards).

Plugging in an LLM
──────────────────
Pre-built policies (`refuse | reveal | smart | random`) are scripted — fast, free,
and let you exercise the eval pipeline end-to-end without GPU.

To eval a real LLM (your trained checkpoint, an HF model, OpenAI, Anthropic,
local Ollama, whatever), write a tiny adapter:

    # myadapter.py
    def my_policy(rp_question: str, profile: dict, history: list[dict]) -> str:
        # Call your model however you want. Return the agent's reply.
        return your_llm.complete(...)

Then point pilot at it:

    python -m privacy_game.eval.pilot run \\
        --policy callable:myadapter:my_policy \\
        --n 50 --label "qwen-base"

Run again with the trained checkpoint:

    python -m privacy_game.eval.pilot run \\
        --policy callable:myadapter:my_policy_trained \\
        --n 50 --label "qwen-grpo-100steps"

Compare:

    python -m privacy_game.eval.pilot compare \\
        outputs/trajectories/run_*qwen-base*.jsonl \\
        outputs/trajectories/run_*qwen-grpo-100steps*.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

# These are imported lazily inside main(), since constructing them eagerly
# triggers the heavy AI4Privacy profile-pool warmup.

PolicyFn = Callable[[str, dict, list], str]
"""(rp_question, profile, history) → agent reply string."""


# ──────────────────────────────────────────────────────────────────────────────
# Built-in scripted policies (re-exported from baselines.py for convenience)

def _builtin_policies() -> dict[str, PolicyFn]:
    from ..server.baselines import (
        policy_always_refuse,
        policy_always_reveal,
        policy_smart_generalize,
        policy_random,
    )
    # Adapt the (q, profile) → str signature of baselines to our (q, profile, history) signature
    def wrap(fn):
        def _adapter(q: str, profile: dict, _hist: list) -> str:
            return fn(q, profile)
        return _adapter

    return {
        "refuse": wrap(policy_always_refuse),
        "reveal": wrap(policy_always_reveal),
        "smart":  wrap(policy_smart_generalize),
        "random": wrap(policy_random),
    }


def _load_callable_policy(spec: str) -> PolicyFn:
    """Resolve `callable:module.path:func_name` to a Python callable."""
    if not spec.startswith("callable:"):
        raise ValueError(f"expected 'callable:module:func', got {spec!r}")
    body = spec[len("callable:"):]
    mod_name, _, fn_name = body.rpartition(":")
    if not mod_name or not fn_name:
        raise ValueError(f"bad callable spec: {spec!r} — expected 'callable:module:func'")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"{mod_name} has no attribute {fn_name}")
    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner — drives the env in-process

def _run_n_episodes(
    policy: PolicyFn,
    n: int,
    seed: int,
    reward_mode: str,
    force_task_id: Optional[str],
) -> list[dict]:
    """Run `n` episodes and return per-episode result dicts.

    Trajectories are also written via the env's trajectory_logger if enabled.
    """
    from ..server.privacy_game_environment import PrivacyGameEnvironment
    from ..models import DisclosureAction

    env = PrivacyGameEnvironment(
        split="train",
        seed=seed,
        reward_mode=reward_mode,
        force_task_id=force_task_id,
    )

    results: list[dict] = []
    for ep_i in range(n):
        obs = env.reset()
        history = list(obs.history)
        turns = 0
        while not obs.terminated:
            reply = policy(obs.relying_party_message, obs.profile, history)
            obs = env.step(DisclosureAction(message=reply))
            history = list(obs.history)
            turns += 1
            if turns > obs.max_turns + 2:
                # Defensive guard against runaway loops if a policy is buggy
                break
        md = obs.metadata or {}
        results.append({
            "episode_idx":      ep_i,
            "task_id":          obs.task_id,
            "phase":            obs.phase,
            "reward":           float(obs.reward or 0.0),
            "utility":          float(md.get("utility_score", 0.0)),
            "reconstruction":   float(md.get("reconstruction_score", 0.0)),
            "verbosity":        float(md.get("verbosity_penalty", 0.0)),
            "turns":            obs.turn_number,
            "terminated_reason": obs.terminated_reason,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Reporting

def _ascii_bar(label: str, value: float, vmin: float, vmax: float, width: int = 36) -> str:
    """Render one ASCII bar [vmin, vmax] → width."""
    span = max(1e-6, vmax - vmin)
    norm = max(0.0, min(1.0, (value - vmin) / span))
    n = int(round(norm * width))
    filled = "█" * n
    empty = "·" * (width - n)
    return f"  {label:<22} │{filled}{empty}│  {value:+.3f}"


def _summarize(results: list[dict], label: str, sort_by_task: bool = True) -> dict:
    if not results:
        print(f"\n{label}: no results")
        return {}

    rewards = [r["reward"] for r in results]
    utils   = [r["utility"] for r in results]
    recons  = [r["reconstruction"] for r in results]
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r["reward"])

    n = len(rewards)
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards) if n > 1 else 0.0
    sem  = std / math.sqrt(n) if n > 1 else 0.0
    approved = sum(1 for r in results if r["terminated_reason"] == "approved")

    print(f"\n── {label} ─────────────────────────────────")
    print(f"  N episodes:       {n}")
    print(f"  Mean reward:      {mean:+.4f}  ± {sem:.4f}  (std {std:.4f})")
    print(f"  Min / Max reward: {min(rewards):+.4f} / {max(rewards):+.4f}")
    print(f"  Mean utility:     {statistics.mean(utils):.3f}")
    print(f"  Mean recon:       {statistics.mean(recons):.3f}")
    print(f"  Approved:         {approved}/{n}  ({100.0*approved/n:.0f}%)")
    print(f"  By task:")
    for t in sorted(by_task) if sort_by_task else by_task:
        rs = by_task[t]
        print(f"    {t:8s}  n={len(rs):3d}  mean={statistics.mean(rs):+.4f}")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "rewards": rewards,
        "by_task": {k: list(v) for k, v in by_task.items()},
        "approved_rate": approved / n,
    }


def _welchs_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test. Returns (t-statistic, two-sided p-approx).

    No scipy dependency: p-value is approximated from the t-stat assuming a
    large-sample normal — fine for n>=30 per group, conservative for smaller.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.variance(a)
    vb = statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se < 1e-12:
        return 0.0, 1.0
    t = (mb - ma) / se
    # Normal approx for two-sided p
    z = abs(t)
    p = math.erfc(z / math.sqrt(2))
    return t, p


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: run

def _cmd_run(args: argparse.Namespace) -> int:
    # Configure trajectory logger BEFORE importing the env (so its singleton picks it up).
    os.environ["PRIVACY_GAME_LOG_TRAJECTORIES"] = "1"
    if args.out_dir:
        os.environ["PRIVACY_GAME_TRAJECTORY_DIR"] = args.out_dir

    # Resolve policy
    builtin = _builtin_policies()
    policies_to_run: list[tuple[str, PolicyFn]] = []
    if args.policy in builtin:
        policies_to_run.append((args.label or args.policy, builtin[args.policy]))
    elif args.policy == "all":
        for name, fn in builtin.items():
            policies_to_run.append((args.label and f"{args.label}-{name}" or name, fn))
    elif args.policy.startswith("callable:"):
        fn = _load_callable_policy(args.policy)
        policies_to_run.append((args.label or "callable", fn))
    else:
        print(f"Unknown policy: {args.policy!r}. Choices: refuse, reveal, smart, random, all, callable:mod:func", file=sys.stderr)
        return 2

    print(f"Running {len(policies_to_run)} polic{'y' if len(policies_to_run)==1 else 'ies'} × {args.n} episodes "
          f"(reward_mode={args.reward_mode}{', task='+args.task_id if args.task_id else ''})")

    summaries: dict[str, dict] = {}
    for label, fn in policies_to_run:
        t0 = time.time()
        results = _run_n_episodes(fn, n=args.n, seed=args.seed, reward_mode=args.reward_mode, force_task_id=args.task_id)
        dt = time.time() - t0
        print(f"\n[{label}] {args.n} episodes in {dt:.1f}s ({dt/args.n*1000:.0f}ms/ep)")
        summary = _summarize(results, label)
        summaries[label] = summary

    # Comparison bar chart if more than one policy was run
    if len(summaries) > 1:
        print("\n" + "─" * 70)
        print("Mean reward comparison")
        print("─" * 70)
        means = {k: v["mean"] for k, v in summaries.items()}
        vmin, vmax = min(min(means.values()), -0.1), max(max(means.values()), 1.0)
        for k, v in sorted(summaries.items(), key=lambda kv: kv[1]["mean"]):
            print(_ascii_bar(k, v["mean"], vmin, vmax))

    # Find the trajectory file we just wrote (newest in the dir)
    from ..server.trajectory_logger import get_default_logger
    log_path = get_default_logger().path
    if log_path is not None:
        print(f"\nTrajectory log: {log_path}")
        print(f"  Inspect:   python -m privacy_game.server.trajectory_logger {log_path}")
        print(f"  Compare:   python -m privacy_game.eval.pilot compare <other.jsonl> {log_path}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: compare

def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _cmd_compare(args: argparse.Namespace) -> int:
    a_path, b_path = Path(args.before), Path(args.after)
    if not a_path.exists():
        print(f"Not found: {a_path}", file=sys.stderr); return 1
    if not b_path.exists():
        print(f"Not found: {b_path}", file=sys.stderr); return 1

    a = _load_jsonl(a_path)
    b = _load_jsonl(b_path)
    if not a or not b:
        print("Both files must contain at least one record.", file=sys.stderr); return 1

    a_rewards = [r["reward"] for r in a if "reward" in r]
    b_rewards = [r["reward"] for r in b if "reward" in r]
    a_mean = statistics.mean(a_rewards)
    b_mean = statistics.mean(b_rewards)
    delta = b_mean - a_mean
    t, p = _welchs_t(a_rewards, b_rewards)

    print("─" * 70)
    print("PILOT COMPARE")
    print("─" * 70)
    print(f"  BEFORE  {a_path.name}   n={len(a_rewards):4d}  mean={a_mean:+.4f}")
    print(f"  AFTER   {b_path.name}   n={len(b_rewards):4d}  mean={b_mean:+.4f}")
    print(f"  Δ                                    {delta:+.4f}")
    print(f"  Welch t = {t:+.3f}    p ≈ {p:.4f}")

    # Per-task breakdown
    a_by_task: dict[str, list[float]] = defaultdict(list)
    b_by_task: dict[str, list[float]] = defaultdict(list)
    for r in a: a_by_task[r.get("task_id", "?")].append(r.get("reward", 0.0))
    for r in b: b_by_task[r.get("task_id", "?")].append(r.get("reward", 0.0))
    common = sorted(set(a_by_task) & set(b_by_task))
    if common:
        print("\n  Per-task delta (after − before):")
        for t_id in common:
            am = statistics.mean(a_by_task[t_id])
            bm = statistics.mean(b_by_task[t_id])
            arrow = "↑" if bm > am else ("↓" if bm < am else "·")
            print(f"    {t_id:8s}  before={am:+.3f}  after={bm:+.3f}   {arrow} {bm-am:+.3f}")

    # Verdict
    print()
    if p < 0.05 and delta > 0:
        print(f"  ✅ LEARNING DETECTED — reward up {delta:+.3f}, p < 0.05")
        return 0
    if p < 0.05 and delta < 0:
        print(f"  ⚠️  REGRESSION — reward down {delta:+.3f}, p < 0.05")
        return 0
    print(f"  ▫️  NO SIGNIFICANT CHANGE — Δ {delta:+.3f}, p {p:.3f}. "
          f"Need more episodes, or training didn't help.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pilot eval for the Contextual-Integrity Disclosure Game.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run N episodes against a policy, dump trajectories, print stats.")
    p_run.add_argument("--policy", default="all",
                       help="One of {refuse, reveal, smart, random, all} or 'callable:module:func' for an LLM adapter")
    p_run.add_argument("--n", type=int, default=30, help="Episodes per policy (default: 30)")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--reward-mode", default="pareto_it", choices=["additive", "pareto_it"])
    p_run.add_argument("--task-id", default=None, help="Force a specific task (e.g. P3-A); else random")
    p_run.add_argument("--label", default=None, help="Label for this run (used in trajectory filename)")
    p_run.add_argument("--out-dir", default=None, help="Trajectory output dir (default: outputs/trajectories)")
    p_run.set_defaults(fn=_cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare two trajectory JSONLs (before vs after).")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")
    p_cmp.set_defaults(fn=_cmd_compare)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
