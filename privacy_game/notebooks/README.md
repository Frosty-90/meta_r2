# Notebooks

## `grpo_train.py`

Colab-compatible script with `# %%` cell markers. Open in Colab or convert to ipynb:

```bash
# Convert to ipynb with jupytext (one-time)
pip install jupytext
jupytext --to notebook grpo_train.py
# Now grpo_train.ipynb exists; upload to Colab.
```

Or simply open `grpo_train.py` in VS Code's Jupyter interactive view (cells are recognized).

## What the script does

1. Pins TRL/transformers/peft/datasets versions
2. Clones the META_H repo into the Colab VM (TODO: replace `<YOUR_USER>` with your GitHub user)
3. Builds the env's Docker image and runs it as a localhost container on port 8000
4. Verifies env round-trip via the Python SDK
5. Defines the TRL wrapper class with one tool: `respond(message)`
6. Constructs `GRPOTrainer` with `environment_factory=DisclosureGameTRL`
7. Smoke run (20 steps) → real run (500 steps)
8. Saves checkpoint + runs eval against scripted baselines
9. Cleanup

## Key choices

- **In-Colab Docker** (not remote HF Space) for training to avoid Cloudflare WebSocket issues
- **Qwen2.5-1.5B-Instruct** as base model (T4-trainable; A100 if HF credits allow it)
- **`max_completion_length=4096`** — counts the FULL multi-turn dialogue, not single agent message. Don't lower this.
- **`num_generations=8`** — GRPO group size; needs `num_generations * batch_size` envs running concurrently. Server's `max_concurrent_envs=8` matches.
- **`temperature=0.8`** — high enough to give group diversity GRPO needs.
