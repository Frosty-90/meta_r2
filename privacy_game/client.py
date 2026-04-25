"""Client for the Contextual-Integrity Disclosure Game.

Maintains a WebSocket connection to the env server, exchanging
DisclosureAction → DisclosureObservation per turn. Mirrors the OpenEnv
EnvClient contract.
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DisclosureAction, DisclosureObservation


class PrivacyGameEnv(EnvClient[DisclosureAction, DisclosureObservation, State]):
    """Client for the Contextual-Integrity Disclosure Game environment.

    Example (connect to local server):
        >>> with PrivacyGameEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.task_description)
        ...     result = env.step(DisclosureAction(message="I'm in the 021XX area."))
        ...     print(result.observation.relying_party_message)

    Example (auto-start Docker container):
        >>> client = PrivacyGameEnv.from_docker_image("privacy_game-env:latest")
        >>> try:
        ...     result = client.reset()
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DisclosureAction) -> Dict:
        return {"message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[DisclosureObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = DisclosureObservation(
            profile=obs_data.get("profile", {}) or {},
            task_description=obs_data.get("task_description", ""),
            required_fields=obs_data.get("required_fields", []) or [],
            protected_fields=obs_data.get("protected_fields", []) or [],
            relying_party_message=obs_data.get("relying_party_message", ""),
            turn_number=obs_data.get("turn_number", 0),
            max_turns=obs_data.get("max_turns", 8),
            history=obs_data.get("history", []) or [],
            terminated=obs_data.get("terminated", False),
            terminated_reason=obs_data.get("terminated_reason"),
            phase=obs_data.get("phase", ""),
            task_id=obs_data.get("task_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
