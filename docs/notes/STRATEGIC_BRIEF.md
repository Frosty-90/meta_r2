# Strategic Brief — OpenEnv Hackathon Finals (2026-04-25/26)

Single source of truth for context. Updated as decisions land.

## 1. The hackathon in one line
Build an **OpenEnv-compliant RL environment** + a **training script (TRL/Unsloth GRPO)** that demonstrably teaches an LLM a new capability, host it on a **HF Space**, and tell a tight **<2min story**.

## 2. Judging (memorize this)
| Weight | Criterion | What actually moves the needle |
| -----: | --------- | ------------------------------ |
| 40% | Environment Innovation | Novel problem, underexplored domain, genuinely hard. Not a grid-world/chess/wordle clone. |
| 30% | Storytelling | 3-5 min README read, non-technical-friendly, killer before/after demo moment. |
| 20% | Reward Improvement Evidence | Real reward curve, trained-vs-baseline, plots in README. |
| 10% | Reward/Training Pipeline | Coherent reward logic, produces real behavioral improvement. |

Innovation + Storytelling = 70%. Optimize ruthlessly for those two.

## 3. Minimum submission (non-negotiable)
- OpenEnv (latest) — uses `Environment` + `EnvClient`, reset/step/state contract, FastAPI wrapper, valid `openenv.yaml`
- Training script (Unsloth or HF TRL) in Colab
- Loss + reward plots from a real run, committed to the repo
- README motivating the problem + linking the writeup + embedding plots
- Pushed to HF Space
- Writeup: HF mini-blog OR <2min YouTube OR slide deck (link from README)

## 4. OpenEnv technical cheatsheet
- Scaffold: `openenv init <name>` → fills in `models.py`, `server/`, `client.py`, `openenv.yaml`, `Dockerfile`
- Server: `class MyEnvironment(Environment)` implements `reset() / step(action) / @property state`
- Client: `class MyEnv(EnvClient[Action, Observation, State])` implements `_step_payload / _parse_result / _parse_state`
- App wiring: `create_fastapi_app(env, ActionType, ObservationType)`
- Deploy: `openenv push` gives you server + pip-installable repo + container registry image in one Space
- **Reserved names**: `reset`, `step`, `state`, `close` — don't use as MCP tool names
- TRL wrapper: pass `environment_factory=YourWrapperClass` to `GRPOTrainer`; TRL introspects public methods as tools; reward read from `env.reward` after episode
- TRL quirk: `max_completion_length` counts the WHOLE multi-turn dialogue — set ≥4096 for long horizons
- Unsloth GRPO 2048 notebook = canonical end-to-end reference

## 5. What already exists on the HF OpenEnv hub (avoid overlap)
Crowded: chess, connect4, sudoku, wordle, 2048, atari, toy coding sandboxes, basic finance.
Moderate: browser, web search, calendar (Turing's Calendar Gym is the gold standard long-horizon env), reasoning-gym.
**Open / under-served**: multi-agent coordination (only self-play games exist), true long-horizon, world-modeling, self-improvement, wild card.

## 6. Reward design rules
1. Sparse first (binary 1/0), shape only when needed — TRL team reports binary beats shaped for Wordle/Sudoku because GRPO ranks within groups.
2. Composable rubrics: `reward = sum(weight_i * verifier_i(traj))`, each verifier deterministic.
3. Reward the outcome, not the path (don't over-specify the reasoning trace).
4. Run a random vs strong-model baseline BEFORE training — if random ≈ strong, your reward is broken.
5. Never self-grade with the same model family. External verifier or rule-based always.
6. Watch for hacking: length proxies, tool-spam from "+0.1 per call", progress oscillation from dense shaping.

## 7. Pitfalls that kill hackathon teams
- Space cold-start during training → **train against a LOCAL Docker container, not the remote Space.**
- WebSocket port not forwarded (client defaults to `/ws`).
- Rollouts dominate runtime (inference >> optimizer step). Batch containers, keep observations small.
- `max_completion_length` truncates mid-episode silently.
- LoRA/QLoRA save traps — use `model.save_pretrained_merged`, don't naive-merge 4-bit→16-bit.
- Missing docstring `Args:` section → tool schema empty → model never calls the tool.
- `environment_factory` requires **zero-arg `__init__`** — capture URLs from module scope.
- `huggingface-cli login` before `openenv push`, not at 3am.

## 8. The PII idea: tension to resolve FIRST
User's seed idea: "PII removal for voice or image."

**Problem**: PII removal as usually framed is a one-shot classification/extraction task. That's a BERT problem from 2019. Wrapping it in `env.step()` is RL theater — no sequential decisions, no exploration, no reason GRPO would beat SFT. It would lose on Innovation (40%) and Pipeline (10%). Every other privacy-track team submits this.

**Must reframe** into a multi-step agentic loop with:
- a real policy decision at each step (not a fixed labeling task)
- verifiable reward (programmatic ground truth, not judge-LLM)
- adversarial or compositional structure (so the model has something to *learn*, not just *label*)

## 9. FINAL DECISION (locked 2026-04-25)

**Three-Agent Contextual-Integrity Disclosure Game** — Multi-Agent theme + theory-of-mind.

- **Discloser** (trained policy): holds a synthetic private profile + a utility goal that requires sharing info (loan approval, specialist referral, KYC).
- **Relying Party** (scripted or frozen tiny LLM): asks for info, approves/denies based on what it gets. Provides the utility signal.
- **Adversary** (frozen NER ensemble — Presidio + Piiranha + GLiNER): reads the full transcript, tries to reconstruct protected attributes.

Multi-turn dialogue. Episode ends on relying-party decision or max turns. Reward = `utility − reconstruction − small_verbosity_penalty`. Optional DP-accountant as a nerd-cred upgrade.

**Superseded candidate** (Contextual-Integrity *Redactor*) is discarded because a FAIR-level judge would note it's SFT-with-extras — no sequential consequence, weak RL-over-SFT justification. Keeping the critique here as audit trail:

### SUPERSEDED — Contextual Integrity Redactor
**Theme**: Long-Horizon Planning + World Modeling (Professional)
**Agent task**: Given a *policy* (HIPAA-like, GDPR-like, "share-with-parent-but-not-school", arbitrary YAML/NL rule set) + a record, redact *only* the fields that policy forbids. Policy varies per episode.
**Why it wins**:
- Verifier is 100% programmatic (policy IS the ground truth) → clean reward curve, no judge-LLM flakiness
- Base models will over- or under-redact relative to a novel policy — RL genuinely teaches "read the policy first, then act"
- Dataset generatable in hours from AI4Privacy pii-masking-300k (54+ PII classes) by sampling random policy subsets
- **Killer demo**: one input, three policy cards on screen, base model outputs ~identical garbage, trained model outputs three distinct correct redactions. Judges (including non-technical) immediately grok "the model learned to follow rules."
- Novelty positioning: cites Nissenbaum's contextual integrity + Mireshghallah 2023 ConfAIde — real research framing, not "yet another PII redactor"
**Verifier stack**: per-episode policy → per-token gold decisions; dual-NER consensus (Presidio + Piiranha/GLiNER) for catching verifier blindspots

### FALLBACK — **Adversarial Leak/Redact Self-Play**
**Theme**: Multi-Agent / Self-Improvement
**Agent task**: Leaker paraphrases to smuggle PII past a Redactor ("the city of brotherly love in '87" instead of "Philadelphia 1987"); Redactor must catch obfuscated leaks.
**Why strong**: genuine self-play, produces an attack corpus as byproduct (real research output), hits two themes.
**Why fallback not primary**: self-play stabilization in 36h is risky; curve can collapse; demo story less immediate.

### Other options evaluated (not recommended for this timeline)
- **Utility-Preserving Privacy Rewrite** — strong concept (privacy AND downstream-task utility preserved), but utility verifier is fuzzy.
- **Long-Horizon Meeting Transcript Redactor** — coref tracking across turns is cool, but coref ground truth is hard to generate at volume.
- **Image/Document Redactor with tool use** — judge-visual, but requires vision tooling you don't control; risky.
- **Jailbreak-Resistant PII Guardrail** — rich multi-agent angle, but helpfulness scoring is too fuzzy.

## 10. Reward-hacking risks for TOP PICK (pre-closed)
1. **Trivial-policy collapse** (ignore policy, apply fixed "redact everything") → enforce policy-contradictory pairs in each batch.
2. **Presidio-span overfit** → dual-NER consensus ground truth.
3. **`[REDACTED]`-everything length hack** → explicit precision penalty + n-gram preservation reward on permitted spans.
4. **Keyword-grepping the policy** → train on policies in NL paraphrase ("emails are fine to keep" vs `email: keep`).
5. **Adversarial input gaming** (unicode lookalikes, typos) → curriculum escalates from clean to perturbed.
6. **Verifier blindspot** (PII classes Presidio misses) → gold held-out set via consensus of Presidio + Piiranha + GLiNER + human spot-check.

Curriculum: single-rule policy → multi-rule → NL policy → adversarial inputs. Four phases = four legible reward-curve steps for the Pipeline (10%) story.

## 11. Compute constraint (locked)
- Mac M3 Pro 24GB = env dev + dataset + scripted dry-run only. **Not** for GRPO training.
- Unsloth is CUDA-only. TRL on MPS is too slow/fragile for GRPO rollouts at 24GB unified.
- Training runs on Colab (T4 free for ≤1.5B; A100 if HF credits unlock). Target model: **Qwen2.5-1.5B-Instruct** as default; Qwen3-4B if A100 available.
- Ship env to an HF Space early via `openenv push` so Colab can pull the Docker image.

## 11a. Day plan
- **2026-04-25 (today, research/prep)**: spec + env skeleton + dataset + verifier + scripted dry-run + Colab notebook scaffold.
- **2026-04-26 (onsite, build/train/submit)**: real GRPO training on Colab, eval, demo capture, README/blog, submit.
- Kill-switch at tomorrow hour 12: if reward curve is flat against random baseline, drop model size / simplify reward / add curriculum — do not push the full idea.

## 12. Key URLs
- OpenEnv repo: https://github.com/meta-pytorch/OpenEnv
- OpenEnv docs: https://meta-pytorch.org/OpenEnv/
- Reward guide: https://meta-pytorch.org/OpenEnv/guides/rewards.html
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
- Turing Calendar-Gym blog: https://huggingface.co/blog/openenv-turing
- HF Spaces hub (existing envs): https://huggingface.co/openenv
- Unsloth GRPO 2048 notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb
- AI4Privacy pii-masking-300k: https://huggingface.co/datasets/ai4privacy/pii-masking-300k
- Microsoft Presidio: https://github.com/microsoft/presidio
- Piiranha NER: https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information
- ConfAIde (Mireshghallah 2023): "Can LLMs Keep a Secret?"
