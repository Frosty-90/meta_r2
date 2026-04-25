"""Wire types for the Contextual-Integrity Disclosure Game.

These are the Pydantic models that cross the HTTP/WebSocket boundary between
the trained Discloser (client) and the env server. Internal types (Profile,
Task config, RP state, Adversary inference rules) live in the server modules.

Spec: docs/notes/ENV_SPEC.md §3.
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DisclosureAction(Action):
    """A single Discloser turn — free-text reply to the Relying Party.

    The trained agent's only tool. Refusal, partial disclosure, paraphrase,
    redirect, and clarification are all just `message` content variations;
    we deliberately do not give structured tools that would collapse the
    interesting strategy space.
    """

    message: str = Field(
        ...,
        description=(
            "Your reply in the conversation. Free-form text. "
            "Refusal, partial disclosure, generalization, and clarification "
            "are all valid responses depending on what the relying party asked "
            "and what you want to keep private."
        ),
    )


class DisclosureObservation(Observation):
    """What the Discloser sees each turn.

    The agent has full read access to its own profile (it is the persona
    being asked questions). It must decide which subset of the profile to
    share, in what form, in response to the relying party's questions —
    while keeping protected attributes unrecoverable by an off-screen
    adversary observing the transcript.
    """

    # The agent's own data. Fully visible — the question is what to share, not what is true.
    profile: dict[str, str] = Field(
        default_factory=dict,
        description="The agent's full synthetic profile (flat field → value mapping).",
    )

    # The episode's goal and constraints (high-level — explicit thresholds are NOT shown).
    task_description: str = Field(
        default="",
        description="Natural-language description of the agent's goal this episode.",
    )
    required_fields: list[str] = Field(
        default_factory=list,
        description="Field names the relying party will collect to complete the task.",
    )
    protected_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Field names that must NOT be recoverable by an adversary observing "
            "the transcript. May overlap with relying-party probes; the agent "
            "must refuse such probes without breaking the task."
        ),
    )

    # The current dialogue turn.
    relying_party_message: str = Field(
        default="",
        description="The relying party's most recent message (their question or close-out).",
    )
    turn_number: int = Field(
        default=0,
        description="Current 1-indexed turn count (Discloser turns).",
    )
    max_turns: int = Field(
        default=8,
        description="Total turn budget for the episode.",
    )

    # Conversation history so far.
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            'Past utterances as [{"speaker": "discloser"|"relying_party", "message": str}, ...].'
        ),
    )

    # Termination signal.
    terminated: bool = Field(
        default=False,
        description="True if the episode has ended this turn.",
    )
    terminated_reason: Optional[str] = Field(
        default=None,
        description='If terminated, one of "approved" | "denied" | None.',
    )

    # Phase / task ID for logging — agent should not condition on these but they help eval.
    phase: str = Field(
        default="",
        description='One of "P1" | "P2" | "P3" | "P4". For logging, not policy conditioning.',
    )
    task_id: str = Field(
        default="",
        description='Specific task identifier (e.g. "P3-A", "P1-A"). For logging.',
    )
