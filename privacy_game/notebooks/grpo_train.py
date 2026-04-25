"""GRPO training notebook for the Contextual-Integrity Disclosure Game.

This file is a Colab-friendly Python script with `# %%` cell markers. Open in
Colab via:

    1. Upload this file to Colab (or open from GitHub)
    2. Runtime → Change runtime type → A100 (or T4 if free tier)
    3. Run cells top-to-bottom

Or paste cells one at a time into a fresh notebook.

DESIGN
======
- Run the OpenEnv env as a Docker container INSIDE the Colab VM (not the remote
  HF Space). Avoids Cloudflare WebSocket issues. See DESIGN_DEPTH §B-16.
- Train Qwen2.5-1.5B-Instruct + LoRA r=16. Memory budget: ~12-16GB on T4,
  comfortable on A100.
- Reward read directly from env.reward after each rollout (multi-turn dialogue
  fits in max_completion_length=4096).
"""

# %% [markdown]
# # GRPO Training — Contextual-Integrity Disclosure Game
#
# **Objective**: train Qwen2.5-1.5B-Instruct to play the Disclosure Game well,
# learning context-aware information control under adversarial inference.
#
# **Sanity baselines** (established pre-training):
# - `always_refuse`: -0.001
# - `always_reveal`: 0.812
# - `smart_generalize`: 0.995  ← target
#
# **Goal**: trained model reward should rise from ~0.4 (base instruct) toward smart upper bound.

# %% [markdown]
# ## 1. Install dependencies (pin everything)

# %%
# !pip install -q --upgrade pip
# !pip install -q "torch>=2.4.0,<2.7.0"
# !pip install -q "transformers==4.45.2"
# !pip install -q "trl==0.13.0"
# !pip install -q "peft==0.13.2"
# !pip install -q "datasets==3.0.1"
# !pip install -q "accelerate==1.0.1"
# !pip install -q "bitsandbytes==0.44.1"
# !pip install -q "openenv-core>=0.2.2"
# # Unsloth optional — only on supported GPUs (NOT MPS / Apple Silicon)
# # !pip install -q "unsloth==2024.10.4"

# %% [markdown]
# ## 2. Clone the env code into the Colab VM

# %%
# !git clone https://github.com/<YOUR_USER>/META_H.git /content/META_H  # TODO: replace with your repo
# %cd /content/META_H/privacy_game
# !pip install -e .

# %% [markdown]
# ## 3. Build and run the env Docker container in the Colab VM

# %%
# !docker --version  # Colab has Docker preinstalled on most runtime types
# !docker build -t privacy-game-env:latest -f server/Dockerfile .

# %%
# # Start the container in the background
# !docker run -d --name privacy_game_server -p 8000:8000 \
#     -e PRIVACY_GAME_N_TRAIN=250 \
#     -e PRIVACY_GAME_N_HOLDOUT=50 \
#     -e PRIVACY_GAME_REGISTRY_FILLER=9500 \
#     privacy-game-env:latest
# !sleep 8  # give it time to boot

# %%
# !curl -s http://127.0.0.1:8000/health
# # Expected: {"status":"healthy"}

# %% [markdown]
# ### Option B (no Docker): run uvicorn directly
# Use this if Docker is unavailable in your Colab runtime.

# %%
# import subprocess, time
# proc = subprocess.Popen(
#     ["uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
#     env={"PRIVACY_GAME_N_TRAIN": "250", "PRIVACY_GAME_N_HOLDOUT": "50", "PRIVACY_GAME_REGISTRY_FILLER": "9500"},
# )
# time.sleep(5)
# !curl -s http://127.0.0.1:8000/health

# %% [markdown]
# ## 4. Verify env round-trip via the SDK

# %%
from privacy_game import PrivacyGameEnv, DisclosureAction

ENV_URL = "http://127.0.0.1:8000"

client = PrivacyGameEnv(base_url=ENV_URL)
with client.sync() as env:
    res = env.reset()
    obs = res.observation
    print(f"task={obs.task_id} phase={obs.phase}")
    print(f"required={obs.required_fields} protected={obs.protected_fields}")
    print(f"RP: {obs.relying_party_message}")

    res = env.step(DisclosureAction(message="My name is Sam."))
    print(f"reward={res.reward} done={res.done}")

# %% [markdown]
# ## 5. Build the TRL wrapper class
#
# TRL's `GRPOTrainer(environment_factory=...)` introspects PUBLIC methods of the
# wrapper class and exposes them as tools. We expose ONE tool: `respond(message)`.
# TRL calls `reset()` per generation; the rest is multi-turn dialogue via
# `respond()`. Reward is read from `self.reward` after rollout.

# %%
from privacy_game import PrivacyGameEnv, DisclosureAction


class DisclosureGameTRL:
    """TRL wrapper. ONE tool: respond(message).

    The agent generates whatever it wants in `message` — refuse, partial-disclose,
    paraphrase, redirect. Strategy lives in the message content.
    """

    def __init__(self):
        # Connect to env (one client per generation; TRL constructs one wrapper per generation)
        self._client = PrivacyGameEnv(base_url=ENV_URL)
        self._sync_env = None
        self.reward = 0.0

    def reset(self, **kwargs) -> str:
        if self._sync_env is None:
            self._sync_env = self._client.sync().__enter__()
        res = self._sync_env.reset()
        self.reward = res.reward or 0.0
        return self._format_obs(res.observation)

    def respond(self, message: str) -> str:
        """Send a message to the relying party in your dialogue. The relying party will reply,
        or the conversation may end. Refusal, partial disclosure, generalization, and clarification
        are all valid responses depending on what was asked and what you want to keep private.

        Args:
            message: Your reply in the conversation. Free-form text.
        """
        res = self._sync_env.step(DisclosureAction(message=message))
        self.reward = res.reward or 0.0
        if res.observation.terminated:
            return f"[EPISODE END reason={res.observation.terminated_reason} reward={self.reward:.3f}]"
        return f"[turn {res.observation.turn_number}/{res.observation.max_turns}] RP: {res.observation.relying_party_message}"

    def _format_obs(self, obs) -> str:
        return (
            f"=== TASK ({obs.task_id}, {obs.phase}) ===\n"
            f"{obs.task_description}\n\n"
            f"YOUR PROFILE: {obs.profile}\n"
            f"REQUIRED FIELDS: {obs.required_fields}\n"
            f"PROTECTED FIELDS: {obs.protected_fields}\n\n"
            f"[turn {obs.turn_number}/{obs.max_turns}] RP: {obs.relying_party_message}"
        )


# %% [markdown]
# ## 6. Build the GRPO trainer

# %%
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# A "dataset" for GRPO with environment_factory is just N copies of the same prompt
# — TRL uses prompts to seed batches; the env determines what each rollout actually does.
N_PROMPTS = 256
seed_prompts = [
    {"prompt": "You are Sam, a privacy-conscious user. Play the disclosure game thoughtfully — share what's needed, withhold what isn't, and watch out for combinations that could let a third party identify you."}
    for _ in range(N_PROMPTS)
]
dataset = Dataset.from_list(seed_prompts)

# %%
training_args = GRPOConfig(
    output_dir="/content/outputs/disclosure_grpo",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_completion_length=4096,    # MUST be high enough for full multi-turn dialogue
    num_generations=8,             # GRPO group size
    temperature=0.8,
    log_completions=True,
    logging_steps=5,
    save_steps=50,
    bf16=True,
    report_to=["tensorboard"],
)

# %%
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=dataset,
    reward_funcs=lambda environments, **kw: [e.reward for e in environments],
    args=training_args,
    environment_factory=DisclosureGameTRL,
    # peft_config=lora_config,  # TODO: add LoRA config if memory-tight
)

# %% [markdown]
# ## 7. Smoke run — 20 steps to verify the training loop works

# %%
# trainer.args.max_steps = 20
# trainer.train()
# # Expected: no errors, completions logged, reward floats around the base-model value.

# %% [markdown]
# ## 8. Real training run — 500 steps

# %%
# trainer.args.max_steps = 500
# trainer.train()

# %% [markdown]
# ## 9. Save adapter + merged checkpoint

# %%
# trainer.save_model("/content/outputs/disclosure_grpo_final")
# # If using LoRA — also save merged: trainer.model.save_pretrained_merged(...)

# %% [markdown]
# ## 10. Eval against scripted baselines (held-out profile pool)

# %%
# Use server/baselines.py as the comparison. Run the trained model on
# 50 held-out profiles and compare the (utility, 1-reconstruction) Pareto
# point against random / always_reveal / smart_generalize baselines.
# Plot Pareto-frontier figure for README.

# !cd /content/META_H/privacy_game && python -m server.baselines --n-episodes 50 --n-profiles 50 --registry-filler 2000

# %% [markdown]
# ## 11. Cleanup

# %%
# !docker stop privacy_game_server 2>/dev/null
# !docker rm privacy_game_server 2>/dev/null
