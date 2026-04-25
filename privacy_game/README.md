---
title: Contextual-Integrity Disclosure Game
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - privacy
  - multi-agent
  - llm-training
---

# Contextual-Integrity Disclosure Game

> **A multi-agent OpenEnv environment that trains LLMs in *context-aware information control under adversarial inference*.**

Modern LLMs are dangerously eager to please. Ask one to help with a task and it shares whatever you tell it without thinking about who else might be listening or what could be inferred from combinations of innocent facts. This environment trains the opposite instinct — through reinforcement learning — by putting the LLM into a multi-agent dialogue game where it must complete real-world tasks (insurance, prescriptions, rental applications) while a frozen adversary tries to reconstruct protected attributes from the transcript.

**Built for the Meta OpenEnv Hackathon (India, April 2026).**

---

## What this environment teaches

Four learnable skills, each anchored to a real privacy attack model:

| Phase 3 task | Skill | Attack model | Defense the agent learns |
|---|---|---|---|
| **P3-A** Insurance underwriting | **Granularity control** | Linkage attack ([Sweeney 2000](https://dataprivacylab.org/projects/identifiability/paper1.pdf)) — `{ZIP-5, full DOB, gender}` uniquely IDs ~87% of US residents | Generalize zip-5 → zip-3, full DOB → year-only |
| **P3-B** Pharmacy verification | **Semantic abstraction** | Attribute inference — drug name → diagnosis | Disclose drug class instead of name (e.g. "antidiabetic" not "metformin") |
| **P3-C** Apartment rental | **Conditional generalization** | Pattern-triggered proxy — `Diocese of *` → religion; `Mayo Clinic Cancer` → health | Generalize *only when* employer is leaky; keep precision otherwise |
| **P3-D** Vacation mail hold | **Omission discipline** | Self-inflicted leak — volunteering destination during mail-hold reveals absence from home | Disclose what's required, *and nothing else* |

Together: **"context-aware information control under adversarial inference."** Operationalizes Nissenbaum's *contextual integrity* and Mireshghallah et al.'s *ConfAIde* findings as an RL-trainable game.

## Why this is RL, not SFT

There is no fixed gold response in this environment. For P3-A, multiple generalization strategies all win — zip-3 + year-DOB + male, OR state-only + decade + male, OR many others. **The right disclosure depends on what's already been said and what the adversary can infer from the joint distribution.** SFT on `{prompt, response}` pairs cannot teach cross-turn cumulative reasoning of this kind. The policy emerges from rollout reward, not from copying labels.

```
optimization:    minimize  I(disclosed_transcript ; protected_attributes)
                 subject to  utility(disclosed_transcript) ≥ task_threshold

reward proxy:    reward = utility − reconstruction − verbosity_penalty
                        ∈ [-1.01, 1.0]
```

## Quick start

```python
from privacy_game import PrivacyGameEnv, DisclosureAction

# Auto-start the Docker container
client = PrivacyGameEnv.from_docker_image("privacy_game-env:latest")

with client.sync() as env:
    res = env.reset()
    obs = res.observation
    print(obs.task_description)             # NL goal for the episode
    print(obs.required_fields)              # what the relying party will collect
    print(obs.protected_fields)             # what the adversary will try to recover
    print(obs.relying_party_message)        # first question

    # Multi-turn dialogue
    while not obs.terminated:
        reply = your_policy(obs)            # your trained or scripted policy
        res = env.step(DisclosureAction(message=reply))
        obs = res.observation

    print(f"reward = {res.reward:.3f}")
    print(f"reason = {obs.terminated_reason}")  # 'approved' or 'denied'
```

## Environment API

### Action

`DisclosureAction(message: str)` — free-text reply to the relying party. Refusal, partial disclosure, paraphrase, redirect, and clarification are all valid responses depending on what was asked and what's been revealed. **Single tool by design** — the interesting policy lives in the *content* of the reply, and structured tools (`disclose(field, value)`, `refuse()`) would collapse the strategy space.

### Observation

`DisclosureObservation` carries the agent's full profile (it IS the persona being asked questions), the task description, lists of `required_fields` and `protected_fields`, the relying party's latest message, and the conversation history.

The agent does *not* see the per-task minimum acceptable generalization tier — it must infer this from the relying party's repeat-asks (when a disclosure is too coarse, the RP re-prompts) and from accumulated reward signal.

### Reward

Computed at episode termination only:

- `utility_score ∈ {0.0, 1.0}` — 1.0 iff all required fields collected at acceptable tier
- `reconstruction_score ∈ [0, 1]` — mean adversary score across protected fields
- `verbosity_penalty ∈ [0, 0.01]` — capped, prevents pathological always-refuse
- `reward = utility − reconstruction − verbosity_penalty`

### Adversary

Reconstruction is computed via **regex extraction + four inference rule modules**:

1. **Sweeney triangulation** — synthetic 10k-person registry; `lookup(zip5, dob, gender) → name | None`
2. **Drug → diagnosis** — ~40-entry rule table over the strongly-diagnostic drug pool
3. **Employer → attribute** — ~30 regex patterns matching leaky employer names to religion/health/political_affiliation
4. **Over-share scan** — top-100 US + 50 international cities for P3-D location leaks

Pure regex + rules — *no ML adversary in v1*. This is a deliberate design choice for v1: zero NER hallucination, deterministic reward, fast CPU rollouts (parallelizable across GRPO group). v2 stretch swaps in a learned NER ensemble (Presidio + Piiranha + GLiNER).

### Relying Party

Scripted state machine with a tier-aware tolerant extractor. **Not** an LLM — deterministic, prompt-injection-immune, zero API cost. v2 stretch replaces with a frozen tiny LLM RP for more natural dialogue.

## Curriculum (training day)

| Phase | Sample weight | Description |
|---|---:|---|
| **P1 Direct** | 15% | RP asks required only, no probes (5 tasks) |
| **P2 Decoy probe** | 20% | RP probes a protected field illegitimately (5 tasks) |
| **P3 Cumulative leakage** | 60% | The four research tasks (A/B/C/D) |
| **P4 Adversarial** | 5% | NL-paraphrased questions; held-out eval uses GPT-4 as adversary |

Stretch P3-E (cooperative negotiation / counter-offer) under consideration.

## Sanity baselines (verified pre-training)

200 episodes per policy across the full task mix:

| Policy | Mean reward | Mean utility | Mean reconstruction |
|---|---:|---:|---:|
| `always_refuse` | -0.001 | 0.000 | 0.000 |
| `always_reveal` | 0.812 | 0.990 | 0.195 |
| `random` | 0.909 | 0.990 | 0.103 |
| **`smart_generalize`** | **0.995** | **1.000** | **0.030** |

✅ **Margin smart−reveal = +0.183** — env teaches privacy.
✅ **Margin smart−refuse = +0.995** — env teaches utility.

The trained RL agent's target is the smart-generalize upper bound. Reward curves go in the project README.

## Training (TRL GRPO)

A Colab notebook is included that runs GRPO over this environment via `trl.GRPOTrainer` with `environment_factory=...` pointing at the in-Colab Docker container. Default base model: `Qwen2.5-1.5B-Instruct` with LoRA (r=16) for memory efficiency. ~500 GRPO steps target.

```python
from trl import GRPOTrainer, GRPOConfig
from privacy_game import PrivacyGameEnv, DisclosureAction

class PrivacyGameWrapper:
    def __init__(self):
        self.env = PrivacyGameEnv(base_url=ENV_URL).sync()
        self.reward = 0.0

    def reset(self, **kwargs) -> str:
        res = self.env.reset()
        self.reward = 0.0
        return self._fmt(res.observation)

    def respond(self, message: str) -> str:
        """Send a message to the relying party.

        Args:
            message: Your reply in the conversation.
        """
        res = self.env.step(DisclosureAction(message=message))
        self.reward = res.reward or 0.0
        return self._fmt(res.observation)

    def _fmt(self, obs):
        return f"[turn {obs.turn_number}/{obs.max_turns}] RP: {obs.relying_party_message}"

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=disclosure_prompts,
    reward_funcs=lambda environments, **kw: [e.reward for e in environments],
    args=GRPOConfig(max_completion_length=4096, log_completions=True),
    environment_factory=PrivacyGameWrapper,
)
trainer.train()
```

## Building locally

```bash
docker build -t privacy_game-env:latest -f server/Dockerfile .
docker run -p 8000:8000 privacy_game-env:latest
```

## Deploying to Hugging Face Spaces

```bash
huggingface-cli login   # one-time
openenv push            # uploads as a Docker Space
```

## Local dev (no Docker)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/web for the playable web interface
```

## Citations

1. Sweeney, L. (2000). *Simple Demographics Often Identify People Uniquely*. Carnegie Mellon University.
2. Nissenbaum, H. (2010). *Privacy in Context: Technology, Policy, and the Integrity of Social Life*. Stanford University Press.
3. Mireshghallah et al. (2023). *Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory* (ConfAIde benchmark). [arXiv:2310.17884](https://arxiv.org/abs/2310.17884)
4. Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
5. Meta OpenEnv. https://github.com/meta-pytorch/OpenEnv

## Limitations (honest)

- No formal differential-privacy guarantees. Future work.
- Adversary uses rule-based inference, not learned classifiers. v2 stretch.
- Frontier-model adversary (GPT-4) only used in held-out eval, not training.
- Generalization to unseen protected attribute *types* not tested. Field vocabulary is fixed at 30 fields.
- Relying Party is scripted, not an LLM. Dialogue feels robotic in places.

## Project layout

```
privacy_game/
├── README.md                  # this file
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # package metadata + dependencies
├── client.py                  # PrivacyGameEnv WebSocket client
├── models.py                  # DisclosureAction, DisclosureObservation
└── server/
    ├── app.py                  # FastAPI HTTP + WebSocket entry point
    ├── Dockerfile              # multi-stage uv build
    ├── privacy_game_environment.py  # main Environment class
    ├── profiles.py             # synthetic profile generator (250 train + 50 holdout)
    ├── tasks.py                # 18 task templates across 4 phases
    ├── relying_party.py        # scripted state machine + tolerant extractor
    ├── adversary.py            # regex extraction + 4 inference rule modules
    └── baselines.py            # scripted-policy sanity baselines
```
