"""Per-episode trajectory logger (JSONL).

Captures one JSON record per *terminated* episode for offline analysis,
reward-curve plotting, and replay. Disabled by default — enable with
`PRIVACY_GAME_LOG_TRAJECTORIES=1`. No overhead when disabled.

Output: one JSONL file per process under `outputs/trajectories/`, named
`run_<UTC-timestamp>_<pid>.jsonl`. Override the directory with
`PRIVACY_GAME_TRAJECTORY_DIR`. Each line is a self-contained episode record:

    {
      "episode_id":       str,            # OpenEnv State episode_id
      "timestamp":        float,          # unix seconds (terminal time)
      "task_id":          str,            # e.g. "P3-A"
      "phase":            str,            # "P1" | "P2" | "P3"
      "reward_mode":      str,            # "additive" | "pareto_it"
      "reward":           float,          # final composed reward
      "utility_score":    float,
      "reconstruction_score": float,
      "verbosity_penalty":    float,
      "terminated_reason":    str,        # "approved" | "denied"
      "n_turns":              int,
      "max_turns":            int,
      "total_agent_tokens":   int,
      "profile":              {...},      # gold profile (sensitive — stays local)
      "extras":               {...},      # per-episode extras (e.g. ticket_count)
      "history":              [{...}],    # full transcript
      "required_fields":      [...],
      "protected_fields":     [...],
      "collected_fields":     {...},      # what RP got + at what tier
      "per_protected_score":  {...},      # leakage per protected field
      "per_protected_recovered": {...},   # actual recovered value (or None)
      "rubric_breakdown":     {...},      # full per-rubric result
    }

Why JSONL: line-delimited, append-only writes are crash-safe; one record per
line lets `jq` / pandas / `wc -l` work without parsing the whole file.

Concurrency: a single module-level logger is shared across env instances in
the same process. A threading.Lock serializes writes so concurrent WebSocket
sessions don't interleave at sub-line granularity. Different processes get
different files (pid-suffixed) so they don't fight over the same file handle.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_DEFAULT_DIR = Path("outputs/trajectories")
_ENV_ENABLE = "PRIVACY_GAME_LOG_TRAJECTORIES"
_ENV_DIR = "PRIVACY_GAME_TRAJECTORY_DIR"


class TrajectoryLogger:
    """Append-only JSONL writer for terminal episodes.

    Construct directly to override defaults, or call `get_default_logger()` to
    use the module-level singleton driven by env vars.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._fh: Optional[Any] = None
        self._path: Optional[Path] = None
        self._records_written: int = 0

        if not enabled:
            return

        if path is None:
            base = Path(os.environ.get(_ENV_DIR, str(_DEFAULT_DIR)))
            base.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = base / f"run_{ts}_pid{os.getpid()}.jsonl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        self._path = path
        # Line-buffered append: every write flushes a complete line, so a
        # crash mid-run never produces a half-record. Append mode is safe
        # if multiple processes target the same path (rare but well-defined
        # on POSIX for line-sized writes).
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def records_written(self) -> int:
        return self._records_written

    def log(self, record: dict) -> None:
        """Append one record. No-op if logger is disabled."""
        if not self._enabled or self._fh is None:
            return
        # default=str — safely serializes anything weird (Decimal, datetime, etc.)
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._records_written += 1

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy)

_default_logger: Optional[TrajectoryLogger] = None
_default_lock = threading.Lock()


def get_default_logger() -> TrajectoryLogger:
    """Return the process-wide default logger, creating it on first call.

    Driven by env vars at first-call time:
        PRIVACY_GAME_LOG_TRAJECTORIES=1   to enable (default: 0 / disabled)
        PRIVACY_GAME_TRAJECTORY_DIR=...   to override output dir
    """
    global _default_logger
    if _default_logger is not None:
        return _default_logger
    with _default_lock:
        if _default_logger is None:
            enabled = _bool_env(_ENV_ENABLE, default=False)
            _default_logger = TrajectoryLogger(enabled=enabled)
    return _default_logger


def reset_default_logger() -> None:
    """Drop the singleton (close its file). Mainly for tests."""
    global _default_logger
    with _default_lock:
        if _default_logger is not None:
            _default_logger.close()
        _default_logger = None


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────────────
# CLI: summarize a trajectory JSONL file
#
#   python -m privacy_game.server.trajectory_logger outputs/trajectories/run_*.jsonl
#
# Prints: total episodes, mean reward, by-task breakdown, by-reason breakdown,
# top-3 highest and lowest rewards.

def _summarize(paths: list[Path]) -> None:
    import statistics
    from collections import Counter, defaultdict

    rewards: list[float] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    by_reason: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    leaderboard: list[tuple[float, str, str]] = []  # (reward, task_id, episode_id)
    n = 0

    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                r = float(rec.get("reward", 0.0))
                rewards.append(r)
                by_task[rec.get("task_id", "?")].append(r)
                by_reason[rec.get("terminated_reason", "?")] += 1
                by_mode[rec.get("reward_mode", "?")] += 1
                leaderboard.append((r, rec.get("task_id", "?"), rec.get("episode_id", "?")[:8]))

    if n == 0:
        print("No records found.")
        return

    print(f"Episodes:          {n}")
    print(f"Reward mean ± std: {statistics.mean(rewards):+.4f} ± {statistics.stdev(rewards) if n > 1 else 0:.4f}")
    print(f"Reward min / max:  {min(rewards):+.4f} / {max(rewards):+.4f}")
    print()
    print("By task (n  ·  mean reward):")
    for task in sorted(by_task):
        rs = by_task[task]
        print(f"  {task:8s}  n={len(rs):3d}  mean={statistics.mean(rs):+.4f}")
    print()
    print(f"By terminated_reason: {dict(by_reason)}")
    print(f"By reward_mode:       {dict(by_mode)}")
    print()
    leaderboard.sort()
    print("Bottom 3:")
    for r, t, eid in leaderboard[:3]:
        print(f"  {r:+.4f}  {t:8s}  {eid}")
    print("Top 3:")
    for r, t, eid in leaderboard[-3:][::-1]:
        print(f"  {r:+.4f}  {t:8s}  {eid}")


if __name__ == "__main__":
    import sys

    args = [Path(p) for p in sys.argv[1:]]
    if not args:
        print(
            "Usage: python -m privacy_game.server.trajectory_logger <file.jsonl> [...]\n"
            "       Summarize one or more trajectory JSONL files.",
            file=sys.stderr,
        )
        sys.exit(2)
    bad = [p for p in args if not p.exists()]
    if bad:
        for p in bad:
            print(f"Not found: {p}", file=sys.stderr)
        sys.exit(1)
    _summarize(args)
