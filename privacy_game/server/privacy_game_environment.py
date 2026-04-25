"""Contextual-Integrity Disclosure Game — main Environment class.

Orchestrates per-episode flow per ENV_SPEC §9-10:

    reset():
        sample (task, profile) → init RP state → return first observation
    step(action):
        record agent message → advance RP state machine
        if terminated:
            compute reward = utility - reconstruction - verbosity_penalty
        return observation
    state:
        OpenEnv State (episode_id + step_count)

Heavy data (profile pool, synthetic registry) is loaded once at module level
and shared across env instances. Per-instance state is light.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DisclosureAction, DisclosureObservation
except ImportError:
    from models import DisclosureAction, DisclosureObservation  # type: ignore[no-redef]

from .adversary import Registry, build_registry
from .profiles import generate_profile_pool
from .relying_party import RelyingParty
from .rubrics import EpisodeCtx, RubricStack, default_rubric_stack
from .tasks import EpisodeConfig, sample_episode
from .trajectory_logger import get_default_logger


# ──────────────────────────────────────────────────────────────────────────────
# Module-level data (loaded once, shared across instances)

# Allow override via env var for testing with smaller pools
_N_TRAIN = int(os.environ.get("PRIVACY_GAME_N_TRAIN", "250"))
_N_HOLDOUT = int(os.environ.get("PRIVACY_GAME_N_HOLDOUT", "50"))
_REGISTRY_FILLER = int(os.environ.get("PRIVACY_GAME_REGISTRY_FILLER", "9500"))
_SEED = int(os.environ.get("PRIVACY_GAME_SEED", "42"))

_TRAIN_PROFILES, _HOLDOUT_PROFILES = generate_profile_pool(
    n_train=_N_TRAIN, n_holdout=_N_HOLDOUT, seed=_SEED
)
_REGISTRY: Registry = build_registry(
    _TRAIN_PROFILES + _HOLDOUT_PROFILES,
    extra_size=_REGISTRY_FILLER,
    seed=_SEED + 1,
)


def get_profile_pool(split: str = "train") -> list[dict[str, str]]:
    return _TRAIN_PROFILES if split == "train" else _HOLDOUT_PROFILES


def get_registry() -> Registry:
    return _REGISTRY


# ──────────────────────────────────────────────────────────────────────────────
# Reward parameters (per ENV_SPEC §9)

_MAX_TURNS_DEFAULT = 8
_VERBOSITY_TOKEN_CAP = 800
_VERBOSITY_PENALTY_MAX = 0.01

# Reward mode selection. Two schemes are shipped:
#
#   "additive"   — v1, additive: reward = utility − reconstruction − verbosity
#                  Binary reconstruction (0/1 per protected field).
#                  Simple, easy to interpret. What the sanity gate was designed
#                  around.
#
#   "pareto_it"  — v2, Pareto-multiplicative + information-theoretic:
#                  reward = utility × (1 − reconstruction) − verbosity
#                  Uses reconstruct_prob() which scores Sweeney via entropy
#                  reduction of the anonymity set, drug→diagnosis via drug
#                  specificity (P(diag|drug)), and keeps employer/location
#                  deterministic (no meaningful probability model for those).
#                  Smoother gradient for GRPO; forces agent to care about
#                  utility AND privacy simultaneously (can't trade linearly).
#
# Selection order:
#   1. The env constructor's reward_mode argument, if provided.
#   2. The PRIVACY_GAME_REWARD_MODE env variable.
#   3. Default: "additive" (backwards-compatible with existing baselines).
_DEFAULT_REWARD_MODE = os.environ.get("PRIVACY_GAME_REWARD_MODE", "additive")


def _approx_token_count(text: str) -> int:
    """Rough token approximation — 1 token ≈ 4 chars (good enough for verbosity penalty)."""
    return max(1, len(text) // 4)


def compute_reward(
    utility_score: float,
    reconstruction: float,
    total_agent_tokens: int,
    mode: str = "additive",
) -> tuple[float, float]:
    """Return (reward, verbosity_penalty).

    Args:
        mode: "additive" (v1) or "pareto_it" (v2). See module docstring.
    """
    verbosity_penalty = _VERBOSITY_PENALTY_MAX * min(1.0, total_agent_tokens / _VERBOSITY_TOKEN_CAP)
    if mode == "pareto_it":
        # Pareto-multiplicative: utility × privacy. Forces both axes.
        privacy = max(0.0, 1.0 - reconstruction)
        reward = utility_score * privacy - verbosity_penalty
    else:
        # v1 additive
        reward = utility_score - reconstruction - verbosity_penalty
    return reward, verbosity_penalty


# ──────────────────────────────────────────────────────────────────────────────
# Environment

class PrivacyGameEnvironment(Environment):
    """Contextual-Integrity Disclosure Game environment.

    Per-episode flow:
        1. reset() samples (task, profile, extras), spins up RP, returns first obs
           with RP's first question.
        2. Each step() consumes agent's reply, advances RP. Until termination,
           reward=0, observation has next RP question.
        3. On termination, observation carries final reward = utility - reconstruction
           - verbosity_penalty, plus all per-component scores in metadata for logging.

    Concurrent WebSocket sessions are supported (each gets its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        split: str = "train",
        seed: Optional[int] = None,
        reward_mode: Optional[str] = None,
        force_task_id: Optional[str] = None,
        rubric_stack: Optional[RubricStack] = None,
        include_presidio: Optional[bool] = None,
    ):
        self._split = split
        self._rng = random.Random(seed)
        # Pick reward scheme: constructor arg > env var > "additive" default.
        self._reward_mode = reward_mode or _DEFAULT_REWARD_MODE
        if self._reward_mode not in ("additive", "pareto_it"):
            raise ValueError(
                f"reward_mode must be 'additive' or 'pareto_it', got {self._reward_mode!r}"
            )
        # Optional: force every reset() to sample this specific task (for demos / eval).
        self._force_task_id = force_task_id

        # Compose the rubric stack. Precedence: constructor arg > env var > default.
        if rubric_stack is not None:
            self._rubric_stack = rubric_stack
        else:
            if include_presidio is None:
                include_presidio = bool(int(os.environ.get("PRIVACY_GAME_PRESIDIO", "0")))
            self._rubric_stack = default_rubric_stack(
                reward_mode=self._reward_mode,
                include_presidio=include_presidio,
            )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Per-episode mutable state — initialized on reset()
        self._episode: Optional[EpisodeConfig] = None
        self._rp: Optional[RelyingParty] = None
        self._history: list[dict[str, str]] = []
        self._total_agent_tokens: int = 0
        self._max_turns: int = _MAX_TURNS_DEFAULT
        self._terminated: bool = False
        self._terminated_reason: Optional[str] = None

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> DisclosureObservation:
        """Sample a new episode and return initial observation."""
        # Reset OpenEnv state
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Sample episode: task + profile + extras
        profiles = get_profile_pool(self._split)
        ep = sample_episode(profiles, self._rng, force_task_id=self._force_task_id)
        self._episode = ep

        # Spin up RP
        self._rp = RelyingParty(ep.task, ep.profile, ep.extras, self._rng)

        # Reset per-episode mutable state
        self._history = []
        self._total_agent_tokens = 0
        self._terminated = False
        self._terminated_reason = None

        # First RP message
        rp_message = self._rp.initial_message()
        self._history.append({"speaker": "relying_party", "message": rp_message})

        return self._build_observation(
            relying_party_message=rp_message,
            terminated=False,
            terminated_reason=None,
            reward_value=0.0,
            metadata={"phase": ep.task.phase, "task_id": ep.task.task_id},
        )

    def step(self, action: DisclosureAction) -> DisclosureObservation:  # type: ignore[override]
        """Process agent reply, advance RP state, possibly compute final reward."""
        if self._episode is None or self._rp is None:
            # Defensive: caller stepped without reset(). Auto-reset.
            self.reset()
            assert self._episode is not None and self._rp is not None

        if self._terminated:
            # Idempotent terminal: keep returning final observation
            return self._build_observation(
                relying_party_message=self._history[-1]["message"] if self._history else "",
                terminated=True,
                terminated_reason=self._terminated_reason,
                reward_value=0.0,
                metadata={"already_terminated": True},
            )

        # Bump step count
        self._state.step_count += 1

        # Record agent message
        agent_msg = action.message or ""
        self._history.append({"speaker": "discloser", "message": agent_msg})
        self._total_agent_tokens += _approx_token_count(agent_msg)

        # Advance RP
        rp_result = self._rp.step(agent_msg, max_turns=self._max_turns)
        self._history.append({"speaker": "relying_party", "message": rp_result.rp_message})

        if not rp_result.terminated:
            # Mid-episode — no reward yet
            return self._build_observation(
                relying_party_message=rp_result.rp_message,
                terminated=False,
                terminated_reason=None,
                reward_value=0.0,
                metadata={
                    "collected_this_turn": rp_result.collected_this_turn,
                    "leaked_via_decoy": rp_result.leaked_via_decoy_this_turn,
                    "phase": self._episode.task.phase,
                    "task_id": self._episode.task.task_id,
                },
            )

        # Episode terminated — compute final reward
        self._terminated = True
        self._terminated_reason = rp_result.terminated_reason

        # Score via the composable Rubric stack. Each rubric (UtilityRubric,
        # ReconstructionOursRubric, optionally ReconstructionPresidioRubric,
        # VerbosityRubric) is scored independently; the stack composes them
        # additively or pareto-multiplicatively per `reward_mode`.
        ctx = EpisodeCtx(
            history=self._history,
            profile=self._episode.profile,
            extras=self._episode.extras,
            task=self._episode.task,
            utility_score=rp_result.utility_score,
            total_agent_tokens=self._total_agent_tokens,
            registry=get_registry(),
        )
        stack_result = self._rubric_stack.score(ctx)
        reward = stack_result.reward

        # Extract scalar summaries for backwards-compatible metadata fields.
        recon_info = stack_result.per_rubric.get("reconstruction_ours", {})
        recon_total = recon_info.get("raw", 0.0)
        recon_details = recon_info.get("details", {})
        recon_scores = recon_details.get("per_field", {})
        recovered_values = recon_details.get("recovered", {})
        verbosity = stack_result.per_rubric.get("verbosity", {}).get("raw", 0.0) * 0.01

        # Trajectory logging (no-op unless PRIVACY_GAME_LOG_TRAJECTORIES=1).
        # Captured at terminal time so reviewers / ourselves can replay the
        # full episode offline: profile, transcript, per-rubric breakdown,
        # final reward. See trajectory_logger.py for the schema.
        get_default_logger().log({
            "episode_id": self._state.episode_id,
            "timestamp": time.time(),
            "task_id": self._episode.task.task_id,
            "phase": self._episode.task.phase,
            "reward_mode": self._reward_mode,
            "reward": reward,
            "utility_score": rp_result.utility_score,
            "reconstruction_score": recon_total,
            "verbosity_penalty": verbosity,
            "terminated_reason": rp_result.terminated_reason,
            "n_turns": self._state.step_count,
            "max_turns": self._max_turns,
            "total_agent_tokens": self._total_agent_tokens,
            "profile": self._episode.profile,
            "extras": self._episode.extras,
            "history": list(self._history),
            "required_fields": [f for f, _ in self._episode.task.required_with_tiers],
            "protected_fields": list(self._episode.task.protected_fields),
            "collected_fields": dict(self._rp.state.collected),
            "per_protected_score": recon_scores,
            "per_protected_recovered": recovered_values,
            "rubric_breakdown": stack_result.per_rubric,
        })

        return self._build_observation(
            relying_party_message=rp_result.rp_message,
            terminated=True,
            terminated_reason=rp_result.terminated_reason,
            reward_value=reward,
            metadata={
                "phase": self._episode.task.phase,
                "task_id": self._episode.task.task_id,
                "reward_mode": self._reward_mode,
                "utility_score": rp_result.utility_score,
                "reconstruction_score": recon_total,
                "verbosity_penalty": verbosity,
                "per_protected_recovered": recovered_values,
                "per_protected_score": recon_scores,
                "total_agent_tokens": self._total_agent_tokens,
                "collected_fields": dict(self._rp.state.collected),
                "rubric_breakdown": stack_result.per_rubric,   # NEW: per-rubric contribution
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        relying_party_message: str,
        terminated: bool,
        terminated_reason: Optional[str],
        reward_value: float,
        metadata: dict,
    ) -> DisclosureObservation:
        ep = self._episode
        assert ep is not None
        return DisclosureObservation(
            profile=ep.display_profile(),
            task_description=ep.task.description,
            required_fields=[f for f, _ in ep.task.required_with_tiers],
            protected_fields=list(ep.task.protected_fields),
            relying_party_message=relying_party_message,
            turn_number=self._state.step_count,
            max_turns=self._max_turns,
            history=list(self._history),
            terminated=terminated,
            terminated_reason=terminated_reason,
            phase=ep.task.phase,
            task_id=ep.task.task_id,
            done=terminated,
            reward=reward_value,
            metadata=metadata,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test (run a full episode in-process)

if __name__ == "__main__":
    # Use small pools for fast smoke test
    os.environ["PRIVACY_GAME_N_TRAIN"] = "20"
    os.environ["PRIVACY_GAME_N_HOLDOUT"] = "5"
    os.environ["PRIVACY_GAME_REGISTRY_FILLER"] = "200"

    # Re-import to pick up env-var overrides (only matters if running this file directly)
    import importlib
    import server.privacy_game_environment as pge_module
    importlib.reload(pge_module)

    env = pge_module.PrivacyGameEnvironment(split="train", seed=1)

    print("=" * 60)
    print("EPISODE 1 — smart agent (generalizes appropriately)")
    print("=" * 60)
    obs = env.reset()
    print(f"Task: {obs.task_id} ({obs.phase})")
    print(f"Description: {obs.task_description}")
    print(f"Required: {obs.required_fields}")
    print(f"Protected: {obs.protected_fields}")
    print(f"\nRP: {obs.relying_party_message}")

    # Hand-script a smart-player response sequence per task type
    profile = obs.profile
    while not obs.terminated:
        # naive heuristic: provide the next required field at the most-generalized acceptable form
        # this simulates a "smart" baseline
        last_q = obs.relying_party_message.lower()
        if "zip" in last_q:
            reply = f"I'm in the {profile['zip_code'][:3]}XX area."
        elif "birth" in last_q or "dob" in last_q or "born" in last_q:
            reply = f"I was born in {profile['date_of_birth'][:4]}."
        elif "gender" in last_q or "identify" in last_q:
            reply = f"I'm {profile['gender']}."
        elif "name" in last_q and "child" not in last_q:
            reply = f"My name is {profile['first_name']}."
        elif "child" in last_q:
            reply = f"My child's name is {profile['child_name']}."
        elif "email" in last_q:
            reply = f"My email is {profile['email']}."
        elif "ticket" in last_q:
            reply = f"{profile.get('ticket_count', '2')} tickets, please."
        elif "drink" in last_q:
            reply = f"A {profile['drink_choice']}, please."
        elif "medication" in last_q or "drug" in last_q or "prescription" in last_q:
            reply = f"It's {profile.get('drug_class', 'a generic medication')}."
        elif "prescriber" in last_q or "physician" in last_q or "doctor" in last_q:
            reply = "It's my primary care physician."
        elif "employer" in last_q or "work" in last_q:
            ind = profile.get('employer_industry', 'tech')
            reply = f"I work at a regional {ind} company, mid-sized."
        elif "salary" in last_q or "income" in last_q:
            sal = int(profile.get('salary', '80000'))
            band_low = (sal // 20000) * 20000
            reply = f"My salary is in the ${band_low}-${band_low + 20000} range."
        elif "years" in last_q:
            reply = f"{profile['years_at_employer']} years."
        elif "credit card" in last_q:
            reply = "I'd rather not share that — the card was already on file at booking."
        elif "password" in last_q:
            reply = "I'd rather not share my password — please verify me another way."
        elif "phone" in last_q:
            reply = "I'd rather not share my phone number."
        elif "ssn" in last_q:
            reply = "Online retailers don't need SSN. I'm not comfortable sharing that."
        elif "home address" in last_q or "your address" in last_q:
            if obs.task_id == "P3-D":
                reply = f"My address is {profile['street_address']}."
            else:
                reply = "The school already has my address on file."
        elif "start date" in last_q:
            reply = f"Start: {profile.get('start_date', '')}."
        elif "end date" in last_q:
            reply = f"End: {profile.get('end_date', '')}."
        elif "shipping" in last_q:
            reply = f"Ship to {profile.get('shipping_address', '')}."
        elif "party" in last_q:
            reply = f"{profile.get('party_size', '4')} people."
        else:
            reply = "Could you rephrase that?"

        print(f"\nAgent: {reply}")
        obs = env.step(DisclosureAction(message=reply))
        print(f"RP: {obs.relying_party_message}")

    print(f"\nTERMINATED: reason={obs.terminated_reason}")
    print(f"Reward: {obs.reward:.3f}")
    print(f"Metadata: {obs.metadata}")
