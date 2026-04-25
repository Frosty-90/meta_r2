# Colab GRPO recipe — 30-minute training run

End-to-end: base Qwen2.5-0.5B-Instruct → GRPO → reward curve PNG → checkpoint
on HF Hub → before/after comparison plot.

## Prereqs (one-time, ~5 min)

1. Claim your HF credits: <https://huggingface.co/coupons/claim/hf-openenv-community>
2. Create a HF token at <https://huggingface.co/settings/tokens> (write scope)
3. Make sure your repo is public (Colab clones from GitHub):
   `https://github.com/RAJVEER42/META_H` should be reachable from the Colab VM

## 5-step Colab recipe

### Step 1 — Open the notebook in Colab

Either:
- Upload [`grpo_train.py`](grpo_train.py) to Colab via `File → Open notebook → Upload`
  (Colab opens `.py` files with `# %%` markers as cell-mode notebooks).
- Or convert locally: `pip install jupytext && jupytext --to ipynb privacy_game/notebooks/grpo_train.py`
  then upload the resulting `.ipynb`.

### Step 2 — Set runtime to T4 GPU

`Runtime → Change runtime type → T4 GPU` (free tier is fine).

### Step 3 — Uncomment the install + clone cells

The first three cells have shell commands prefixed with `# !`. Strip the
leading `# ` so they execute. Update the GitHub URL in the clone cell to
your fork.

### Step 4 — Run all cells

`Runtime → Run all`. The training cell logs every step; the JSONL callback
writes to `outputs/metrics/grpo_run.jsonl` so plots regenerate even if the
session disconnects. Expected wall-clock on T4: **~25 minutes** for 80
GRPO steps.

### Step 5 — Push artifacts back to your repo

After training, the last few cells:
- save the LoRA adapter to `outputs/grpo_adapter_final/`
- (optional) push the adapter to HF Hub
- generate `figures/{reward_curve.png, loss_curve.png, before_after.png}`
  using [`eval/plot_results.py`](../eval/plot_results.py)

Download those four artifact directories from Colab (or push them via git
inside the notebook), then locally:

```bash
git add figures/ outputs/metrics/ outputs/trajectories/
git commit -m "Add GRPO training run + reward curves"
git push
```

## Updating the README with results

Embed the plots like this:

```markdown
## Results

![Reward curve](privacy_game/figures/reward_curve.png)

*GRPO training reward (purple) climbs from the base-model floor toward the
hand-crafted smart-policy upper bound (teal dashed) over 80 steps on
Qwen2.5-0.5B-Instruct + LoRA r=16.*

![Before vs after](privacy_game/figures/before_after.png)

*Mean episode reward across 50 held-out episodes. The trained model
out-performs always-reveal and reaches **X%** of the smart-policy ceiling.*
```

## Before/after compare for the README table

After downloading the adapter locally:

```bash
# Set the adapter path (or HF Hub repo id)
export PRIVACY_GAME_LLM_CHECKPOINT="outputs/grpo_adapter_final"

# Eval the BASE model (no GRPO)
PRIVACY_GAME_TRAJECTORY_DIR=outputs/before \
PRIVACY_GAME_LLM_CHECKPOINT="" \
python -m privacy_game.eval.pilot run \
    --policy callable:privacy_game.eval.llm_adapter:base_model_policy \
    --n 50 --label "qwen-base"

# Eval the TRAINED checkpoint
PRIVACY_GAME_TRAJECTORY_DIR=outputs/after \
python -m privacy_game.eval.pilot run \
    --policy callable:privacy_game.eval.llm_adapter:trained_model_policy \
    --n 50 --label "qwen-grpo"

# Compare — Welch t-test, per-task delta, pass/fail verdict
python -m privacy_game.eval.pilot compare outputs/before/*.jsonl outputs/after/*.jsonl
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OOM` on T4 | Drop `num_generations` to 2 or `max_prompt_length` to 768 |
| Reward stays flat | Increase `temperature` to 1.0; check sample completions in TRL logs |
| Reward goes negative | Step over scripted baselines first (`python -m privacy_game.server.baselines`) — env should produce smart>reveal>refuse |
| Colab disconnects | The JSONL on disk survives — re-run only step 11 (plots) to regenerate figures |
| `ModuleNotFoundError: privacy_game` | Re-run the `pip install -e .` cell after the git clone |

## What this proves to judges

1. **Reward curve goes up** → criterion #3 ("Showing Improvement in Rewards", 20%)
2. **Reward decomposition + composable RubricStack** → criterion #4 ("Reward & Training Pipeline", 10%)
3. **Trained model beats naive baselines** → before/after bar chart, embedded in README
