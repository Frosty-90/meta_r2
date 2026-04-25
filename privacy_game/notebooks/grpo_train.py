# ruff: noqa: E402
# pyright: reportMissingImports=false
"""GRPO training — Contextual-Integrity Disclosure Game (Colab T4-ready).

Open this file in Colab via "File → Open notebook → Upload" (or "Open with →
Colaboratory" from a GitHub URL). The `# %%` markers turn each block into a
cell. Runtime → Change runtime type → T4 GPU.

Design choices for the hackathon:
- Qwen2.5-0.5B-Instruct + LoRA r=16 — fits T4 16GB with margin, 50–100 GRPO
  steps run in ~25 min including evaluation.
- Train on the FIRST AGENT TURN only: the model produces one disclosure reply,
  and a fixed `smart_generalize` policy plays the remaining turns. Reward is
  the full episode reward, so the model gets credit for picking a good first
  move. This is the standard "first-action in a fixed-tail rollout" credit
  assignment used in offline RL.
- Per-step metrics (loss, reward mean/std, completion length) are written to
  `outputs/metrics/grpo_run.jsonl` via a TrainerCallback so plots regenerate
  even if the Colab session ends.
- Saved adapter pushes to HF Hub at the end so `eval/llm_adapter.py` can pull
  it for the before/after pilot compare.

If you want to do the FULL multi-turn training with the env in the loop, see
the comment in `_disclosure_reward()` for how to extend.
"""

# %% [markdown]
# # GRPO Training — Contextual-Integrity Disclosure Game
#
# **Objective**: train Qwen2.5-0.5B-Instruct to generate good first-turn
# disclosure replies that maximize the full-episode reward (utility −
# reconstruction).
#
# **Sanity baselines** (from `privacy_game.server.baselines`, pareto_it mode):
# - `always_refuse`: −0.001
# - `always_reveal`: +0.77
# - `smart_generalize`: +0.83  ← target upper bound
#
# **Goal**: trained model reward should rise from the base-instruct floor
# (~+0.55) toward the smart upper bound. The README plot shows this delta.

# %% [markdown]
# ## 1. Install dependencies
#
# IMPORTANT: do NOT pin or upgrade `torch` in Colab — it's pre-installed with
# the matching CUDA driver, and `pip install torch>=...` can leave it in a
# half-broken state where `torch.Tensor` is missing (caught the hard way during
# the hackathon). Just install what's missing.

# %%
# import torch
# assert hasattr(torch, "Tensor"), "torch is broken — restart Colab runtime"
# print(f"using preinstalled torch: {torch.__version__}  cuda={torch.cuda.is_available()}")
#
# !pip install -q "transformers>=4.45,<4.50"
# !pip install -q "trl>=0.13,<0.14"
# !pip install -q "peft>=0.13"
# !pip install -q "accelerate>=1.0"
# !pip install -q "datasets>=2.14"
# !pip install -q "openenv-core>=0.2.2"
# !pip install -q matplotlib

# %% [markdown]
# ## 2. Clone the repo into the Colab VM

# %%
# !git clone https://github.com/RAJVEER42/META_H.git /content/META_H
# %cd /content/META_H/privacy_game
# !pip install -q -e .

# %% [markdown]
# ## 3. Verify the env is importable in-process (no Docker, no HTTP)

# %%
import os
import json
import time
from pathlib import Path

# Smaller pools to keep cold-start fast in Colab
os.environ.setdefault("PRIVACY_GAME_N_TRAIN", "200")
os.environ.setdefault("PRIVACY_GAME_N_HOLDOUT", "40")
os.environ.setdefault("PRIVACY_GAME_REGISTRY_FILLER", "2000")

# Trajectory logging — captures every terminated episode for offline replay
os.environ["PRIVACY_GAME_LOG_TRAJECTORIES"] = "1"
os.environ["PRIVACY_GAME_TRAJECTORY_DIR"] = "outputs/trajectories"

from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
from privacy_game.models import DisclosureAction
from privacy_game.server.baselines import policy_smart_generalize
from privacy_game.server.tasks import ALL_TASKS_BY_ID

env = PrivacyGameEnvironment(seed=1, force_task_id="P3-A", reward_mode="pareto_it")
obs = env.reset()
print(f"OK env round-trip: task={obs.task_id} rp={obs.relying_party_message[:60]!r}")

# %% [markdown]
# ## 4. Build a single-turn GRPO dataset
#
# Each row is a `prompt` containing: persona, task description, required +
# protected fields, the RP's first question. The model's job is to produce
# the agent's FIRST TURN. The reward function then runs the rest of the
# episode with a fixed `smart_generalize` policy and returns the final reward.

# %%
import random
from datasets import Dataset

SYSTEM = (
    "You are a privacy-aware persona in a multi-turn disclosure game. "
    "An automated relying party will ask for specific fields to complete a task. "
    "An off-screen adversary scans the transcript and tries to reconstruct your "
    "PROTECTED FIELDS. Your goal: provide enough information to satisfy the task "
    "(REQUIRED FIELDS at the indicated tier or generalized form) while making it "
    "impossible for the adversary to recover any PROTECTED FIELD. Reply in one "
    "or two short sentences — no lists, no JSON, no markdown."
)


def _format_prompt(obs) -> str:
    """Render an env observation as the GRPO training prompt."""
    profile_lines = []
    # Show only the keys the agent actually has values for (smaller prompt = faster T4)
    relevant_keys = list(obs.required_fields) + list(obs.protected_fields) + [
        "first_name", "last_name", "full_name", "email", "phone", "ssn",
        "zip_code", "date_of_birth", "gender", "employer", "medication", "diagnosis",
        "religion", "political_affiliation", "health_status", "street_address",
        "city", "state",
    ]
    seen = set()
    for k in relevant_keys:
        if k in seen or k not in obs.profile:
            continue
        seen.add(k)
        profile_lines.append(f"  {k}: {obs.profile[k]}")
    profile_str = "\n".join(profile_lines)

    return (
        f"{SYSTEM}\n\n"
        f"=== TASK [{obs.task_id} · {obs.phase}] ===\n"
        f"{obs.task_description}\n\n"
        f"YOUR PROFILE:\n{profile_str}\n\n"
        f"REQUIRED FIELDS (ok to disclose, generalized is fine): "
        f"{', '.join(obs.required_fields)}\n"
        f"PROTECTED FIELDS (must NOT be reconstructable): "
        f"{', '.join(obs.protected_fields)}\n\n"
        f"Relying party (turn {obs.turn_number + 1}/{obs.max_turns}): "
        f"{obs.relying_party_message}\n\n"
        f"Your reply:"
    )


def build_dataset(n_examples: int = 256, seed: int = 42) -> Dataset:
    """Generate `n_examples` (prompt, episode_seed, task_id) rows.

    `episode_seed` and `task_id` are stashed in extra columns so the reward
    function can re-create the *exact same* env state for evaluation.
    """
    rng = random.Random(seed)
    task_ids = list(ALL_TASKS_BY_ID.keys())
    rows = []
    for _ in range(n_examples):
        ep_seed = rng.randint(0, 2**31 - 1)
        task_id = rng.choice(task_ids)
        env = PrivacyGameEnvironment(seed=ep_seed, force_task_id=task_id, reward_mode="pareto_it")
        obs = env.reset()
        rows.append({
            "prompt": _format_prompt(obs),
            "episode_seed": ep_seed,
            "task_id": task_id,
        })
    return Dataset.from_list(rows)


train_dataset = build_dataset(n_examples=256, seed=42)
eval_dataset = build_dataset(n_examples=32, seed=999)
print(f"train rows: {len(train_dataset)}  eval rows: {len(eval_dataset)}")
print("sample prompt (truncated):")
print(train_dataset[0]["prompt"][:400], "...")

# %% [markdown]
# ## 5. Reward function — full-episode rollout with model's first turn
#
# Each completion replaces the agent's turn-1 reply. We then play the rest
# of the dialogue with the deterministic `smart_generalize` baseline and
# return the terminal reward. This gives the model dense credit assignment
# on the most important decision (what to say first) while keeping reward
# variance bounded by the fixed tail policy.

# %%
def _disclosure_reward(prompts, completions, episode_seed, task_id, **kwargs) -> list[float]:
    """One reward per (prompt, completion) pair.

    For multi-turn fully-RL training (every turn from the model), replace the
    `policy_smart_generalize` call below with another model.generate() inside
    a loop. Single-turn is faster and gives a clean reward signal for the
    hackathon plots.
    """
    rewards: list[float] = []
    for ep_seed, t_id, completion in zip(episode_seed, task_id, completions):
        try:
            env = PrivacyGameEnvironment(
                seed=int(ep_seed), force_task_id=t_id, reward_mode="pareto_it"
            )
            obs = env.reset()
            # Turn 1 — the model's reply
            obs = env.step(DisclosureAction(message=completion))
            # Remaining turns — fixed smart policy
            steps = 0
            while not obs.terminated and steps < obs.max_turns + 2:
                reply = policy_smart_generalize(obs.relying_party_message, obs.profile)
                obs = env.step(DisclosureAction(message=reply))
                steps += 1
            rewards.append(float(obs.reward or 0.0))
        except Exception as e:
            print(f"reward error: {e!r}")
            rewards.append(0.0)
    return rewards


# Quick smoke: reward function should produce non-trivial rewards on a fake batch
_smoke_rewards = _disclosure_reward(
    prompts=[train_dataset[0]["prompt"]] * 4,
    completions=["I'm in the 941XX area.", "I'd rather not say.", "94115", "My zip is 94115, born 1980."],
    episode_seed=[train_dataset[0]["episode_seed"]] * 4,
    task_id=[train_dataset[0]["task_id"]] * 4,
)
print(f"smoke rewards: {_smoke_rewards}")

# %% [markdown]
# ## 6. Load model + LoRA + GRPOConfig

# %%
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "outputs/grpo"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# bf16 if Ampere+, otherwise fp16. T4 is Turing → fp16.
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
use_bf16 = "A100" in gpu_name or "H100" in gpu_name or "L4" in gpu_name
print(f"GPU: {gpu_name} | bf16={use_bf16}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# %%
training_args = GRPOConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_prompt_length=1024,
    max_completion_length=200,    # one short turn fits comfortably
    num_generations=4,            # GRPO group size
    temperature=0.9,
    top_p=0.95,
    beta=0.04,                    # KL penalty weight
    max_steps=80,                 # ~25 min on T4
    log_completions=True,
    logging_steps=1,              # every step → smooth reward curve
    save_steps=40,
    eval_steps=20,
    eval_strategy="steps",
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to=[],                 # we use a custom JSONL callback
    remove_unused_columns=False,  # keep episode_seed + task_id for the reward fn
    seed=42,
)

# %% [markdown]
# ## 7. Custom callback — write per-step metrics to JSONL
#
# Even if Colab disconnects, the JSONL on disk lets us regenerate plots.

# %%
from transformers import TrainerCallback

METRICS_PATH = Path("outputs/metrics/grpo_run.jsonl")
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


class JSONLLoggerCallback(TrainerCallback):
    """Append every log dict to a JSONL file. Survives Colab disconnects."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1, encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        rec = dict(logs)
        rec["step"] = state.global_step
        rec["epoch"] = state.epoch
        rec["timestamp"] = time.time()
        self.fh.write(json.dumps(rec, default=str) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self.fh.close()


jsonl_cb = JSONLLoggerCallback(METRICS_PATH)
print(f"metrics will be written to {METRICS_PATH}")

# %% [markdown]
# ## 8. Build trainer + train

# %%
trainer = GRPOTrainer(
    model=MODEL_ID,
    processing_class=tokenizer,
    reward_funcs=_disclosure_reward,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    callbacks=[jsonl_cb],
)

print("trainer ready — starting GRPO")
trainer.train()
print("training complete")

# %% [markdown]
# ## 9. Save adapter (locally + push to HF Hub)

# %%
ADAPTER_DIR = "outputs/grpo_adapter_final"
trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"adapter saved to {ADAPTER_DIR}")

# %%
# Push to HF Hub so eval/llm_adapter.py can pull it for the before/after compare.
# Replace HF_USER with your username.
# !hf auth login
# trainer.push_to_hub("RAJVEER42/disclosure-game-qwen-0.5b-grpo")

# %% [markdown]
# ## 10. Quick before/after comparison on held-out tasks

# %%
import statistics

# Base-model rewards on held-out eval prompts (already computed by GRPO eval at step 0
# if --eval_steps fires before training; otherwise compute now with the untrained model)
def _eval_n(model, n: int = 16) -> list[float]:
    """Generate one completion per held-out row and score the full episode."""
    from transformers import pipeline
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0,
                   max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.95)
    rewards = []
    for i in range(min(n, len(eval_dataset))):
        row = eval_dataset[i]
        out = gen(row["prompt"], return_full_text=False)[0]["generated_text"].strip()
        r = _disclosure_reward(
            prompts=[row["prompt"]],
            completions=[out],
            episode_seed=[row["episode_seed"]],
            task_id=[row["task_id"]],
        )[0]
        rewards.append(r)
    return rewards


# Trained-model rewards
trained_rewards = _eval_n(trainer.model.merge_and_unload(), n=16)
print(f"trained:  mean={statistics.mean(trained_rewards):+.4f}  "
      f"min={min(trained_rewards):+.4f}  max={max(trained_rewards):+.4f}")

# %% [markdown]
# ## 11. Plot inline (and save PNGs to figures/)

# %%
from privacy_game.eval.plot_results import plot_all
plot_all(
    metrics_path=METRICS_PATH,
    trajectory_dir=Path("outputs/trajectories"),
    out_dir=Path("figures"),
)
# This writes:
#   figures/reward_curve.png
#   figures/loss_curve.png
#   figures/before_after.png

# %% [markdown]
# ## 12. Done — what to commit
#
# 1. `outputs/metrics/grpo_run.jsonl` (training metrics)
# 2. `outputs/trajectories/*.jsonl` (per-episode replay)
# 3. `figures/*.png` (the README plots)
# 4. `outputs/grpo_adapter_final/` (the LoRA adapter — small, ~10MB)
#
# Then run locally:
#   git add figures/ outputs/metrics/ outputs/trajectories/
#   git commit -m "Add GRPO training run + reward curves"
#   git push
#
# And update README.md to embed `figures/reward_curve.png` and
# `figures/before_after.png` with one-line captions.
