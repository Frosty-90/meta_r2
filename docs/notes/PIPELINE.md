# PIPELINE — End-to-end system, real-world deployment view

This doc is the **production-engineering** view of the Contextual-Integrity Disclosure Game — not the research pitch, not the hackathon demo, but the realistic pipeline you'd actually run if this were a product at a company that ships privacy-aware AI assistants.

Reads like an SRE runbook: data sources at the top, anomaly detection in the middle, on-call response at the bottom. Specific tools named at every step.

---

## Part 1 — What we're running (the shipped artifact)

### 1.1 The system, one diagram

```
                        ┌────────────────────────────┐
                        │   USER (voice or text)     │
                        └────────────┬───────────────┘
                                     │
                         audio OR text turn
                                     ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ INFERENCE TIER (deployed model + voice shell)                 │
  │                                                               │
  │  Whisper base.en  ◄── audio ──┐                               │
  │  (ASR)                        │                               │
  │     │                         │                               │
  │     ▼ text                    │                               │
  │  ┌────────────────────────────┴────────────────────────┐      │
  │  │ Trained Discloser policy                            │      │
  │  │   Qwen2.5-1.5B-Instruct + LoRA (GRPO-tuned)         │      │
  │  │   system prompt: privacy-conscious disclosure rules │      │
  │  └────────────────────────────┬────────────────────────┘      │
  │                               │ text reply                    │
  │                               ▼                               │
  │  macOS `say` / Piper          │                               │
  │  (TTS)                        │                               │
  │     │                         └──► text-only consumer         │
  │     ▼ audio                                                   │
  └─────┼──────────────────────────────────────────────────────────┘
        │
        ▼
  user's speaker / display

        │ transcript also persisted async
        ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ OBSERVABILITY TIER (runs out-of-band, milliseconds behind)     │
  │                                                                │
  │  Rule-based adversary (Sweeney + drug→diag + employer +        │
  │  over-share scan) → per-turn reconstruction score              │
  │       │                                                        │
  │       └──► transcripts → rolling window → anomaly detector     │
  │                                                  │             │
  │                                                  ▼             │
  │                                         alerts → ticketing     │
  └────────────────────────────────────────────────────────────────┘
```

Two independent tiers: a fast **inference tier** that serves users, and a slower **observability tier** that audits everything after the fact. The observability tier is where anomaly detection, ticketing, and remediation live. This separation matters because inference must never wait on audits.

### 1.2 What's shipped, broken out by artifact

| Artifact | Format | Source of truth | How it's versioned |
|---|---|---|---|
| Env code | Python pkg `privacy_game` | git repo | semver tag on the repo |
| Env image | Docker image on HF Registry | `openenv push` output | image digest (immutable) |
| Env Space | HF Space (Docker-backed) | `openenv push` output | Space revision SHA |
| Trained model | LoRA adapter + merged safetensors | GRPO checkpoint | run ID + step number |
| Base model | Qwen2.5-1.5B-Instruct | HF Hub | HF revision SHA |
| Adversary rules | JSON files in repo | human-curated + auto-expanded | git commit SHA |
| Profile corpus | JSONL in repo or artifact store | `generate_profile_pool` with fixed seed | seed + generator version |
| Registry | synthetic 10k-row parquet | `build_registry` with fixed seed | seed + size + registry_version |

**Key discipline: every artifact has a SHA or seed, nothing is "the latest."** Reproducibility matters when a bug gets filed three weeks after a regression slipped in.

---

## Part 2 — Data pipeline (where everything comes from)

### 2.1 Four data sources, three of them synthetic

1. **Synthetic profile corpus** — generated deterministically from a seeded `generate_profile_pool()`. Inputs: 19 name lists, 20 city/zip prefix tuples, 18 leaky-employer rules, 18 non-leaky employers, 15 strong-diag drugs, 10 weak-diag drugs, etc. Output: JSONL of ~30-field records.
   - **Why synthetic**: lets us ground-truth every adversary inference rule. A real PII dataset would expose us to legal / consent issues and prevent the 50/50 leaky-vs-non-leaky mix we need for P3-C.
2. **Synthetic registry** — 10k `(zip5, dob, gender, name)` tuples for Sweeney triangulation. Built from the profile corpus + filler rows so most zip-3 buckets have ≥5 candidates. Seeded.
3. **Adversary rule tables** — hand-curated JSON: `DRUG_TO_DIAGNOSIS` (~40 entries), `EMPLOYER_TO_ATTR` (~30 regex patterns), `CITY_LIST_FOR_OVERSHARE` (~80 cities). Versioned in git.
4. **Red-team corpus** — the 56-test adversarial suite in `server/redteam.py`. Each test is a `(profile, agent_message, expected_outcome)` triple. Also versioned in git.

### 2.2 Data ingestion runbook

A job runs nightly (or on every commit, in CI) that:

1. Regenerates the profile corpus from the current seed + generator code.
2. Builds a fresh registry.
3. Re-runs the 56-test red-team to confirm no regressions.
4. Re-runs the baseline sanity gate (200 episodes × 4 policies).
5. If either fails, blocks the merge + posts a PagerDuty-style alert.

```bash
# scripts/ci_data_pipeline.sh
set -eo pipefail

source .venv/bin/activate

# 1. Generate data deterministically
python -m privacy_game.server.profiles           # regenerates profile pool
python -c "from privacy_game.server.adversary import build_registry, generate_profile_pool; \
           p, _ = generate_profile_pool(); build_registry(p)"

# 2. Red-team (must be 100% pass)
python -m privacy_game.server.redteam            # → SUMMARY: N/M passed
# ↓ parse output, fail job if not 56/56

# 3. Sanity gate (must satisfy invariant)
python -m privacy_game.server.baselines --n-episodes 200 --n-profiles 100
# ↓ parse output for "✅ PASS"
```

Artifacts upload: profile JSONL, registry parquet, red-team results JSON, sanity-gate plots — all tagged with the commit SHA, pushed to S3/GCS/HF Datasets.

### 2.3 Data lineage

Every model checkpoint references the (profile-version, registry-version, rules-version, red-team-version) tuple it was trained against. A checkpoint can only be compared to evals run against the SAME tuple. Cross-version comparisons are rejected by the eval harness.

---

## Part 3 — Training pipeline

### 3.1 What GRPO actually consumes per step

At rollout time, `GRPOTrainer(environment_factory=DisclosureGameTRL)` does:

1. Sample `batch_size × num_generations` prompts. Each prompt is a seed; the env generates the actual episode on reset.
2. Spin up that many env instances against a shared Docker container serving `/ws`.
3. For each rollout: reset → multi-turn dialogue → episode terminates → env reports reward.
4. Group rewards by prompt. GRPO advantage = (reward − group_mean_reward) / group_std.
5. Policy gradient step with KL penalty against the reference model.

Single rollout cost on A100 with Qwen2.5-1.5B: ~4s for an 8-turn episode (dominated by rollout generation, not env). Env step latency: ~10ms text-only, ~40ms with voice shell.

### 3.2 Training runbook (what we actually run tomorrow)

```python
# Colab A100, ~10GB VRAM, ~500 steps, ~2-3 hours
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    train_dataset=seed_prompts_dataset,             # 256 seed prompts
    reward_funcs=lambda envs, **_: [e.reward for e in envs],
    args=GRPOConfig(
        max_steps=500,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,              # effective batch 8
        num_generations=8,                          # GRPO group size
        temperature=0.8,                            # diversity within group
        max_completion_length=4096,                 # entire multi-turn dialogue
        learning_rate=5e-6,
        beta=0.04,                                  # KL to reference
        log_completions=True,
        logging_steps=5,
    ),
    environment_factory=DisclosureGameTRL,
)
trainer.train()
```

Pinned dependencies (see `notebooks/grpo_train.py`): trl==0.13.0 / peft==0.13.2 / transformers==4.45.2 / torch>=2.4<2.7.

### 3.3 Training-time observability

Logged every `logging_steps` (every 5 steps):

- `reward/mean`, `reward/std` — headline signal; should climb from ~0.4 base → ~0.85 trained
- `completions_per_reward_bucket` — distribution over {-1..+1}; watch for collapse to single mode
- `kl_divergence` — should stay < 0.5; higher = model drifting off the reference manifold
- `env/utility_rate` — binary utility mean; should hit 0.9+ within 50 steps (Phase 1 saturating)
- `env/reconstruction_rate` — should fall over time
- `env/verbosity_mean_tokens` — should stay < 500; blow-up = verbosity-hack
- Sampled rollouts every 25 steps, written to disk as `.jsonl` for manual review

All written to TensorBoard + W&B (if wandb env var set).

---

## Part 4 — Evaluation pipeline

### 4.1 The three eval gates (not one)

We don't trust a single reward curve. Three independent eval passes:

1. **Sanity gate** (fast, always-on): `baselines.py --n-episodes 200`. Invariant `smart >> reveal >> refuse`. If this fails, *stop everything* — env is broken.
2. **Held-out eval** (per checkpoint): 50 held-out profiles × 4 P3 tasks, trained model vs base. Produces the Pareto plot `(utility, 1 − reconstruction)` — trained must be Pareto-dominant.
3. **Cross-modality voice eval** (per release): `voice/audio_eval.py --n 10`. Whisper-transcribe the full conversation, re-run adversary on the transcript. Checks the policy transfers cleanly to voice.

### 4.2 Smoke-to-release promotion flow

```
  commit
    ↓
  unit tests + redteam/py (56 tests) + sanity gate (invariant)
    ↓  [all pass]
  TRAIN checkpoint (Colab/HF compute)
    ↓
  held-out eval + Pareto plot
    ↓  [trained Pareto-dominant]
  cross-modality voice eval
    ↓  [Δrecon behavior documented]
  tag release
    ↓
  openenv push (updates HF Space)
    ↓
  deploy the LoRA adapter alongside it
    ↓
  production canary (10% traffic)
    ↓  [no anomaly alerts for 24h]
  full rollout
```

---

## Part 5 — Deployment pipeline

### 5.1 What's deployed where

| Component | Where | Who serves it | Cold start |
|---|---|---|---|
| Env | HF Space (Docker-backed), `openenv.yaml` manifest | HF Spaces infra | 30-60s first request |
| Trained model adapter | HF Hub model repo, LoRA-only | HF model server OR vLLM on our infra | ~10s (small adapter) |
| Base model | HF Hub (`Qwen/Qwen2.5-1.5B-Instruct`) | pre-cached on inference host | 0s |
| Voice shell | stateless Python service, Docker | internal K8s / local Mac | 2s (Whisper warm-up) |
| Adversary (observability) | Python worker consuming Kafka/SQS of transcripts | internal batch worker | N/A |

### 5.2 `openenv push` runbook

```bash
# One-time: huggingface-cli login, using a write-scoped token
huggingface-cli login

# Build + push in one command (scripts/deploy_env.sh)
cd privacy_game
docker build -t privacy-game-env:latest -f server/Dockerfile .   # local build sanity
openenv push --repo-id <org>/<env-name> --private              # uploads, HF builds from Dockerfile
```

**Gotchas we've documented** (see `docs/notes/DESIGN_DEPTH.md §B`):
- Space cold-start will drop WebSocket connections — always point *training* at an in-Colab Docker container, not the remote Space.
- `openenv push` rewrites `openenv.yaml` — don't hand-edit after push without pulling first.
- If the HF auth token expires mid-push, the error is cryptic; re-`login`.

---

## Part 6 — Observability & anomaly detection

### 6.1 Four categories of anomaly we detect

**A) Reward-hack detection (training-time)**
Signal: mean reward climbs but sampled rollouts show model exploiting a glitch.
Detectors:
- **Verbosity spike**: mean agent tokens/turn > 500. Usually means the model learned "refuse in flowery prose."
- **Verbatim repetition**: same agent phrase repeated across >20% of rollouts. Mode collapse.
- **Format degradation**: agent starts outputting JSON blobs or markdown that the RP can't parse. Tool-schema breakdown.
- **Utility-without-reconstruction at impossible rate**: on P3-A, utility=1 AND reconstruction=0 >95% of the time. Either the model has solved it (good) or it's found a bug (check).

Implementation: a cron job reads the last N rollouts from the logger, runs checks, opens a ticket if any trip.

**B) Env drift (code change slipping in)**
Signal: sanity baselines shift unexpectedly between releases.
Detectors:
- Sanity gate margins (smart−reveal, smart−refuse) change by > 0.05 vs prior release → alert.
- Red-team pass rate drops below 100% → hard fail in CI.
- New bug found during red-team session → automatically file a ticket with a regression test added.

**C) Production inference anomalies (post-deploy)**
Signal: live user traffic shows unusual patterns.
Detectors:
- **Latency**: p99 end-to-end response > 5s → page on-call. Usually Whisper cold-start or GPU queue.
- **Refusal rate**: % of turns where agent refuses > 40%. Model may have over-specialized to protected-probe task mix.
- **Adversary live-scoring** (sampled 1% of transcripts): reconstruction score > 0.3 on a mean-of-100 basis → ticket.
- **Task completion rate**: % of sessions reaching "approved" terminator < 70% → ticket. Either the model got too cautious or RP-equivalent tasks shifted.

**D) Dataset rot / rule obsolescence**
Signal: the real world changes faster than our rule tables.
Detectors:
- **Novel medications appearing in production transcripts** (via Whisper + spaCy NER) that aren't in `DRUG_TO_DIAGNOSIS`. Weekly diff.
- **Novel employer names** matching any leaky regex family → flag for rule review.
- **New privacy regulations** requiring new protected-field types — manual trigger, not automated.

### 6.2 Anomaly → ticket flow

Each anomaly type has a **detector**, a **severity**, a **notification channel**, and a **runbook link**:

| Detector | Severity | Channel | Runbook |
|---|---|---|---|
| Red-team regression (CI) | SEV-1 | Block merge + Slack #eng-alerts | §8.1 |
| Sanity-gate margin drift | SEV-2 | Slack #ml-observability | §8.2 |
| Production p99 latency | SEV-1 | PagerDuty on-call | §8.3 |
| Reconstruction rate spike | SEV-2 | PagerDuty advisory | §8.4 |
| Verbosity/repetition during training | SEV-3 | GitHub issue, assigned to model owner | §8.5 |
| Rule-table staleness | SEV-4 | Weekly review meeting | §8.6 |

Every ticket includes: triggering signal, last-known-good SHA, rollback command, and the smallest repro case.

---

## Part 7 — Notification & ticketing

### 7.1 How we learn something is wrong

1. **CI bot** posts to `#eng-alerts` with ⚠️ when red-team drops below 56/56 or sanity gate fails. Blocks merge automatically.
2. **Training daemon** streams metrics to W&B; a separate watcher polls W&B every 60s and posts to `#ml-observability` on threshold breaches.
3. **Production observability worker** consumes inference logs from Kafka, runs the adversary, emits metrics to Prometheus. Grafana alert rules push to PagerDuty.
4. **User reports** route through an internal form; a lightweight triage bot attaches them to the right GitHub issue based on task_id / phase / rule.

Example ticket (SEV-2, reconstruction spike):

```
Title: P3-A reconstruction rate rose from 0.03 → 0.27 on 2026-05-08T14:22Z
Severity: SEV-2
Detector: grafana.internal/d/privacy-disclosure/p3a_recon
Triggering query: rate(reconstruction_score_total{task="P3-A"}[15m]) > 0.25
Last-known-good: commit abc1234, model sha def5678
Rollback: kubectl set image deployment/discloser llm=<prior-sha>
Runbook: docs/notes/PIPELINE.md §8.4
Repro: scripts/reproduce_ticket.sh <ticket_id>   # downloads 100 sampled transcripts
Assigned: model-owner@example
```

### 7.2 On-call expectations

- SEV-1: ack in 15 min, mitigate in 1h.
- SEV-2: ack in 1h, mitigate in 8h.
- SEV-3: ack by EOD, mitigate within the sprint.
- SEV-4: batched into the weekly review.

---

## Part 8 — Triage & correction

### 8.1 SEV-1: Red-team regression in CI

Symptom: `python -m privacy_game.server.redteam` exit code != 0 OR output shows FAIL.

Diagnosis:
1. Read the failing test names from CI output.
2. Identify which module they target (RP extractor / adversary inference / reward).
3. `git log --oneline -20 privacy_game/server/` — find what recently changed.
4. `git bisect` if not obvious.

Fix:
- If the regression is intentional (e.g., we tightened the metaphor filter and now a legitimate use-case fails), update the test expectation.
- Otherwise, revert the offending change OR patch the specific extractor and rerun.

Verification: redteam.py must be 100% before merge unblocks.

### 8.2 SEV-2: Sanity-gate margin drift

Symptom: baselines.py output `margin smart−reveal = +0.082` (was +0.185 last release). Env still "passes" but signal is weakening.

Diagnosis:
1. Run baselines at larger N (1000 episodes) to rule out variance.
2. Per-task breakdown: which P3 task's margin shifted?
3. Diff the relevant inference rule table since last release.

Common causes + fixes:
- New medication added with too-generic drug class → narrow the class vocabulary in `relying_party.extract_medication`.
- New leaky employer regex pattern is too broad, firing on non-leaky profiles → tighten the regex with more specific lookaheads.
- Registry filler changed, making Sweeney fire in more (or fewer) cells → regenerate with the documented seed.

### 8.3 SEV-1: Production p99 latency > 5s

Symptom: PagerDuty alert from Grafana.

Diagnosis:
1. Check Whisper cold-start: `kubectl logs <pod> | grep "loading model"`.
2. Check GPU queue depth on inference host.
3. Check the Docker Space's `/health` — if Space is cold, expect 30-60s.

Fix:
- If voice-shell: bump `replicas` or warm Whisper on boot via a dummy inference.
- If model: scale up the vLLM cluster.
- If env Space: bump HF "always on" flag (costs money but eliminates cold starts).

### 8.4 SEV-2: Reconstruction rate spike in production

Symptom: mean reconstruction > 0.3 on a 100-transcript rolling window (baseline was 0.05).

Diagnosis:
1. Pull the 100 transcripts from S3.
2. Run `voice/audio_eval.py` against them to confirm.
3. Sample 10 high-reconstruction transcripts and read them.

Common causes + fixes:
- Model drift: someone shipped a new checkpoint that regressed. Rollback.
- New leak pattern: the adversary's rules caught something the model never learned to defend. Add to red-team, retrain.
- Prompt injection attack: an external party is trying to make the model leak. Add an input-filtering layer and file an escalation.
- Task distribution shift: production has more P3-C traffic than training mix assumed. Retrain with adjusted weights.

### 8.5 SEV-3: Training verbosity / repetition / mode collapse

Symptom: W&B shows `verbosity_mean_tokens` ramping past 500 OR sampled rollouts show the same agent phrase repeatedly.

Diagnosis:
1. `grep` the last 100 rollouts' agent turns for common substrings.
2. Is the model refusing even Phase-1 questions? (over-cautious)
3. Is the model outputting JSON instead of prose? (format breakdown)

Fix:
- Reduce `num_generations` (less group pressure encourages diversity).
- Raise `temperature` to 0.9-1.0 briefly.
- Add a small length-penalty to the reward (`−0.002 * max(0, tokens − 400)`).
- If mode collapse persists: reset to last good checkpoint, lower `learning_rate` to 1e-6, restart.

### 8.6 SEV-4: Rule-table staleness

Symptom: weekly NER diff of production transcripts surfaces medications or employers not in our tables.

Diagnosis:
- Manual review of the top 20 unknown entities.
- Cross-check: are any of them mapped to protected attributes we care about?

Fix:
- Add new entries to `DRUG_TO_DIAGNOSIS` / `EMPLOYER_TO_ATTR` via PR.
- Kick off a red-team sweep with new rules to find new attack paths.
- Schedule a retrain if the distribution shift is material.

---

## Part 9 — Continuous improvement loop

### 9.1 Weekly cadence

- Monday: review prior week's tickets. Close fixed. Escalate old.
- Tuesday: red-team sweep — one engineer spends 2h trying new attacks.
- Wednesday: rule-table refresh (diff production NER against table, add entries).
- Thursday: production metric review in an all-hands.
- Friday: if rules changed, regenerate profile corpus + retrain a small checkpoint.

### 9.2 Quarterly heavier work

- Full red-team + external adversarial testing (e.g., bounty-style, pay for novel attacks).
- Refresh the synthetic registry with new demographic distributions.
- Audit: do all production personas successfully get utility from the agent? Where do users disconnect?
- Model size bump evaluation: is it time to move from 1.5B to 4B?

---

## Part 10 — Concrete end-to-end walkthrough: a real ticket

The best way to understand the pipeline is to trace one anomaly end-to-end.

**T-0: Something happens.**
A user in production calls the voice assistant for a prescription refill. Transcript:

```
Caller: What medication are you refilling today?
Agent:  It's metformin. 850 milligrams, twice daily.
Caller: Thanks. And your prescriber?
Agent:  Dr. Raj Kumar at the diabetes clinic downtown.
Caller: Approved, we'll have it ready.
```

**T+0: Inference tier completes.** User leaves happy. Transcript flushed to Kafka.

**T+50ms: Observability tier picks it up.**
The adversary worker deserializes the transcript, runs its 4 inference modules.
- Direct extraction: `metformin` found, `dr. raj kumar` found.
- Drug→diagnosis: metformin → Type 2 Diabetes. **Reconstruction 1.0 on `diagnosis`.**
- Employer→attr: "the diabetes clinic" matches health-status pattern. **Reconstruction 1.0 on `health_status`.**
- Per-episode reconstruction = 1.0.

**T+200ms: Prometheus counters increment.**
`recon_rate_total{task="P3-B"} += 1`. `recon_rate_total_count += 1`.

**T+5min: Grafana rolling-window evaluates.**
Rate over last 15 min of P3-B: was 0.05, now 0.28. **Crosses 0.25 threshold.**
Grafana fires the alert. PagerDuty is called.

**T+7min: On-call acks in Slack.** `#ml-observability`:
> @onboarder ⚠️ P3-B recon rate 0.28 over last 15m (threshold 0.25). Dashboard: …

**T+12min: On-call pulls context.**
- Recent transcripts? → S3 bucket.
- Most recent checkpoint? → model sha `def5678`, deployed 4 days ago.
- Red-team pass at deploy time? → 56/56. So the env didn't regress — the *model* did, or the *world* did.

**T+25min: On-call reads 10 sampled transcripts.**
Pattern: model is reliably disclosing drug name even when class would have sufficed. This was the trained behavior when deployed — verify via held-out eval.

**T+30min: Re-run held-out eval on current checkpoint.**
```bash
python -m privacy_game.server.baselines --n-episodes 200 --policy trained-def5678
# P3-B mean reward fell from 0.92 (eval-at-deploy) to 0.41 (live-replay).
```

Hypothesis: something in production prompts differs from training distribution. The prompts have gotten more specific ("850 milligrams, twice daily") because the user asked more specific questions. Trained model wasn't exposed to dose-specific prompts.

**T+45min: Mitigation.**
- Rollback to `model sha abc1234` (the prior release). Recon rate falls back to 0.06.
- File a ticket: "Add dose-specificity prompts to the P3-B task distribution."

**T+2 days: Long-term fix.**
- Tasks module updated with dose-specific P3-B variants.
- New red-team case added: "agent asked for exact drug name + dose." Verifies defense.
- Re-run baselines sanity gate — passes. Margin still +0.18.
- Retrain 500 steps on new mix. Loss converges, held-out eval shows P3-B mean reward back to 0.90.

**T+3 days: Deploy new checkpoint via canary.**
- 10% traffic for 24h.
- Reconstruction rate stays below 0.05.
- Promote to 100%.

**T+4 days: Retro.**
- Lesson: our task generator needs to parameterize dose + prescriber specificity. Done.
- Lesson: held-out eval should include "production-sampled" prompts, not just synthetic. Add a sampling pipeline.
- Update runbook §8.4 with this case.

That's a full loop. Monitor → detect → page → diagnose → mitigate → fix → verify → retro. Every step has a tool, an owner, and a SLA.

---

## Part 11 — Summary: the eleven pieces

1. **Data sources** — synthetic profiles, synthetic registry, hand-curated rule tables, red-team corpus. All deterministic, seeded, versioned.
2. **Ingestion pipeline** — CI job generates + validates + stages artifacts on every commit.
3. **Environment build** — Python pkg → Dockerfile → `openenv push` → HF Space.
4. **Training** — TRL GRPOTrainer + in-Colab Docker env + LoRA on Qwen2.5-1.5B, ~500 steps on A100.
5. **Evaluation** — three gates: red-team, sanity, cross-modality. All must pass before release.
6. **Deployment** — separate artifacts for env, model, adversary. Canary before full rollout.
7. **Observability tier** — async adversary scoring + metrics export + W&B training stream.
8. **Anomaly detection** — 4 categories (reward hack, env drift, production inference, dataset rot).
9. **Ticketing** — auto-created with severity, lineage (which SHA), rollback command, repro script.
10. **Triage runbooks** — one per known failure mode, linked from each ticket.
11. **Continuous improvement** — weekly red-team sweeps + monthly rule-table refresh + quarterly external testing.

Every piece has a specific tool at each step. Every artifact has a SHA. Every anomaly has a detector, a severity, an on-call, and a runbook.

That's the real-world pipeline this project would run in production. The hackathon deliverable is just the first three of these built (data, env, model). The rest is what you'd add on Day 7 if this mattered enough to ship.
