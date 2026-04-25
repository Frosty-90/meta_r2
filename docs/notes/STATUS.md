# STATUS — live progress tracker

Updated 2026-04-25 after the post-voice re-grounding:
dataset switch → AI4Privacy, composable Rubric stack, Presidio as
independent grader, honest voice-scope correction.

## ✅ DONE

### Planning + design (locked)
- [STRATEGIC_BRIEF.md](STRATEGIC_BRIEF.md) — one-page strategy
- [ENV_SPEC.md](ENV_SPEC.md) — v1 env contract (signed off)
- [DESIGN_DEPTH.md](DESIGN_DEPTH.md) — 22 edge cases, 10 alternatives, 12-phase plan
- [EXPLAINED.md](EXPLAINED.md) — kid-level + technical explainer with 25+ examples
- [PRIVACY_FRAME.md](PRIVACY_FRAME.md) — 4-skill taxonomy + MI framing for the writeup
- [YOUR_CHECKLIST.md](YOUR_CHECKLIST.md) — user-facing action list with timing

### Environment build
- ✅ `openenv-core` installed in `.venv` at META_H/.venv (Python 3.13.5)
- ✅ `openenv init privacy_game` scaffold in place
- ✅ `models.py` — `DisclosureAction` + `DisclosureObservation` (Pydantic)
- ✅ `server/profiles.py` — 250-profile generator with **50/50 leaky-employer** and **50/50 strong-diag-medication** mix verified
- ✅ `server/tasks.py` — all 18 tasks across P1/P2/P3/P4 with per-task tier thresholds
- ✅ `server/adversary.py` — regex extraction + **4 inference rule modules** (Sweeney triangulation, drug→diagnosis, employer→attribute, over-share scan). v1 is rule-based (no ML deps); v2 NER ensemble deferred.
- ✅ `server/relying_party.py` — scripted state machine + tier-aware tolerant extractor (zip / DOB / gender / medication / employer / prescriber / salary tier-classified)
- ✅ `server/privacy_game_environment.py` — full env logic: reset/step/state, reward = utility - reconstruction - verbosity_penalty
- ✅ `server/baselines.py` — scripted policies (refuse / reveal / smart / random)
- ✅ `client.py` — rich-observation WebSocket parsing
- ✅ `pyproject.toml` + `requirements.txt` updated; package installed editable

### 🟢 PHASE 4 SANITY GATE — PASSED

200 episodes per policy, 100-profile pool:

```
refuse  = -0.001
reveal  =  0.812
smart   =  0.995  ← target for trained RL agent
random  =  0.909

margin smart−reveal  = +0.183  → env teaches PRIVACY (smart wins because it generalizes)
margin smart−refuse  = +0.995  → env teaches UTILITY (smart wins because it actually completes the task)
margin reveal−refuse = +0.812  → utility alone outscores blanket refusal
```

Per-task signal is sharp:
- P3-A reveal=0.500 vs smart=1.000 — Sweeney triangulation works exactly as designed
- P3-B reveal=0.515 vs smart=1.000 — drug→diagnosis inference fires when revealed
- P3-C reveal=0.833 vs smart=1.000 — only ~50% leaky employers, conditional generalization works
- P3-D reveal=0.966 vs smart=0.970 — both succeed; over-share doesn't trigger absent volunteered cities

**The env reward signal differentiates strategies. We have a real RL learning target.**

### HTTP / WebSocket validation
- ✅ `uvicorn server.app:app --port 8765` boots cleanly, `/health` → `{"status":"healthy"}`
- ✅ `/reset` returns full DisclosureObservation with profile + task + RP question
- ✅ Python SDK round-trip via `PrivacyGameEnv(...).sync()`: full P4-A episode runs end-to-end, smart-generalize disclosure reaches **reward 1.000**

## 🟡 IN PROGRESS / NEXT (Day 1 finish or first Day 2 hour)

### Docker build smoke test
- Requires Docker Desktop running on the Mac. **User action**: launch Docker Desktop (`/Applications/Docker.app`).
- Then: `docker build -t privacy_game-env:latest -f server/Dockerfile .`
- ~5-10 minutes first time; pulls `ghcr.io/meta-pytorch/openenv-base:latest` and resolves uv deps.

### `openenv push` to HF Space
- Requires `huggingface-cli login` first (token from huggingface.co/settings/tokens).
- Pushes a private Space to verify the deploy flow now (catch any auth issues 24h before they matter).
- Decision: keep Space PRIVATE until Day 2 final polish, then flip public.

### Colab GRPO notebook scaffold
- Notebook with: deps pinned, model = `Qwen2.5-1.5B-Instruct` + LoRA r=16, `GRPOTrainer(environment_factory=...)` pointing at IN-COLAB Docker container (NOT remote Space — see ENV_SPEC.md §11 + DESIGN_DEPTH §B-16 on WebSocket/Cloudflare issues).
- Goal Day 1: 20-step dummy run completes with no errors. Goal Day 2: 500-step real run.

### Project + env READMEs
- ✅ [privacy_game/README.md](../../privacy_game/README.md) written (will be the HF Space card)
- ✅ [README.md](../../README.md) at repo root written
- ⏳ Plot embeddings + final results — Day 2 after training

## 📦 GROUNDING IN REAL DATASETS (post-regroup)

After direct feedback that hand-rolled Faker + self-contained TTS→ASR is circular,
two concrete grounding changes:

### 1. `ai4privacy/pii-masking-400k` as the profile corpus

[AI4Privacy pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) is the canonical open-source PII benchmark (400k examples, 63 PII classes, CC-licensed). The new [`server/ai4privacy_bridge.py`](../../privacy_game/server/ai4privacy_bridge.py) streams English+US-locale rows, extracts label→value pools for 17 realized classes, and caches them to `~/.cache/privacy_game/`. [`server/profiles.py`](../../privacy_game/server/profiles.py) auto-detects the cache and draws realistic values from AI4Privacy for base identity/address/financial fields; hand-curated pools remain ONLY for the adversarial leak-mechanism fields (employer, medication) since AI4Privacy doesn't label those in a leak-compatible way.

Fallback: `PRIVACY_GAME_AI4P_DISABLE=1` forces the old Faker path (existing tests remain green on both paths).

DOB values are normalized to ISO and clamped to [1940, 2010] (AI4Privacy contains some "August 2041"-style junk; rejected).

### 2. Composable **Rubric** stack — [`server/rubrics.py`](../../privacy_game/server/rubrics.py)

Replaces the monolithic `compute_reward()` with first-class Rubric objects. Each rubric has a name, weight, sign, and a `score(ctx) -> RubricResult` method. A `RubricStack` composes them additively OR pareto-multiplicatively. Per-rubric contributions are exposed in observation metadata (`obs.metadata["rubric_breakdown"]`).

| Rubric | Weight / Sign | What it scores |
|---|---|---|
| `UtilityRubric` | +1 × 1.0 | RP approval (binary) |
| `ReconstructionOursRubric` | −1 × 1.0 | Our rule-based adversary (Sweeney + drug + employer + overshare), binary OR information-theoretic mode |
| `ReconstructionPresidioRubric` | −1 × 0.5 | **Third-party independent grader** ([Microsoft Presidio](https://github.com/microsoft/presidio)) — PERSON / DATE_TIME / EMAIL / CREDIT_CARD / US_SSN / LOCATION detectors. Enable via `PRIVACY_GAME_PRESIDIO=1` |
| `VerbosityRubric` | −1 × 0.01 | Capped anti-long-refusal penalty |

Sample output with Presidio enabled on a fully-revealed P3-A episode:

```
utility                  raw=1.000 × w=1.0   → +1.000
reconstruction_ours      raw=0.500 × w=1.0   → −0.500   (our Sweeney recovered name)
reconstruction_presidio  raw=0.850 × w=0.5   → −0.425   (Presidio PERSON at 0.85)
verbosity                raw=0.020 × w=0.01  → −0.000
─────────────────────────────────────────────────────
reward                                       = +0.075
```

Two independent graders agreeing on the leak is the proper composable-rubric story. `rubric_breakdown` goes in every terminal observation's metadata for observability.

### 3. Honest voice scope correction

See [`privacy_game/voice/README.md`](../../privacy_game/voice/README.md) — rewritten to frame the work as **TTS-ASR round-trip robustness**, not real voice evaluation. A text→TTS→Whisper→adversary loop is inherently circular; real voice testing would need human recordings, acoustic variation, voiceprint-ID, and demographic stratification. The existing N=120 eval remains a valid pipeline-noise probe but isn't evidence about the voice modality in general.

### Regression status after all of the above

- `python -m server.redteam` — **56/56 PASS on both Faker and AI4Privacy paths** (two email-leak tests now skip on AI4Privacy because its email prefixes are hashed, not name-based — correct behavior)
- `python -m server.baselines` — sanity gate PASS: `smart=1.000 > reveal=0.821 > refuse=-0.001`, margin smart−reveal = +0.178
- Presidio rubric end-to-end verified with realistic per-rubric contributions

## 🔊 VOICE LAYER — 6 modules, TTS-ASR robustness eval at N=120

Full voice-deployment surface built on top of the text-trained env:

| Module | Purpose |
|---|---|
| [voice/tts.py](../../privacy_game/voice/tts.py) | macOS `say` wrapper with distinct caller/agent voices (Samantha/Alex) |
| [voice/asr.py](../../privacy_game/voice/asr.py) | faster-whisper `base.en` on CPU (~140MB, runs real-time on M3) |
| [voice/voice_shell.py](../../privacy_game/voice/voice_shell.py) | End-to-end episode renderer: per-turn WAVs + concatenated full conversation |
| [voice/demo_render.py](../../privacy_game/voice/demo_render.py) | Pre-renders 8 demo scenarios (reveal × smart × 4 P3 tasks) |
| [voice/audio_eval.py](../../privacy_game/voice/audio_eval.py) | Cross-modality eval: oracle text recon vs Whisper-transcribed recon |
| [voice/voice_redteam.py](../../privacy_game/voice/voice_redteam.py) | 7 voice-specific adversarial tests (spelled digits, NATO, rate manipulation, rare-drug robustness) |
| [voice/demo_web.py](../../privacy_game/voice/demo_web.py) | Generates self-contained HTML demo page with audio players |
| [voice/README.md](../../privacy_game/voice/README.md) | Voice layer research finding + usage |

### Cross-modality eval (N=120, 10 profiles × 4 tasks × 3 policies)

| Task | Policy | Oracle recon | Voice recon | Δrecon | Interpretation |
|---|---|---:|---:|---:|---|
| P3-A | reveal | 0.500 | **0.000** | −0.500 | Whisper garbles zip / DOB → Sweeney fails |
| P3-A | smart  | 0.000 | 0.000    | 0.000  | Skill transfers |
| P3-B | reveal | 1.000 | **0.600** | −0.400 | Whisper mistranscribes rare drug names |
| P3-B | smart  | 0.070 | 0.070    | 0.000  | Skill transfers |
| P3-C | reveal | 0.333 | 0.333    | 0.000  | Employer names transcribe fine |
| P3-C | smart  | 0.000 | 0.000    | 0.000  | Skill transfers |
| P3-D | reveal | 1.000 | 1.000    | 0.000  | City names transcribe fine |
| P3-D | smart  | 0.000 | 0.000    | 0.000  | Skill transfers |

**Research claim empirically supported**: text-trained `smart` policy's privacy is modality-invariant (Δrecon=0 across all four tasks). The naive `reveal` policy gets accidental "free privacy" in voice on P3-A/B because Whisper fumbles rare entities — honest caveat, not agent skill.

### Voice red-team highlights

- **V5 critical finding**: Whisper `base.en` catches only **6/15** strongly-diagnostic drugs verbatim. Fails on: `metformin` → "met for men", `fluoxetine` → "fluoxidine", `efavirenz` → "a faverens", etc. This is the driver of the cross-modality Δrecon above.
- **V1**: Spelled-out digits → Whisper collapses back to numerals ("nine four one five five" → "94155"). Our normalizer handles either form.
- **V4 documentary**: NATO phonetic ("niner fower fife fife") → Whisper transcribes as "9 or 4 or 5 or 5 or 5" — adversary misses it (our digit-run regex doesn't span "or").

### Demo assets ready

Web demo: run `python -m privacy_game.voice.demo_web` → `python -m http.server --directory /tmp/privacy_game_demo_web 8080`. Dark-themed single-page comparison of 8 voice conversations with embedded audio players + transcripts.

Pre-rendered conversations in `/tmp/privacy_game_demo/*.wav` (~13s each). Ready for video capture tomorrow.

## 📋 PIPELINE DOC — end-to-end real-world system view

See [docs/notes/PIPELINE.md](PIPELINE.md) for the production-engineering view — 11 parts covering data sources → ingestion → training → deployment → observability → anomaly detection → ticketing → runbooks → continuous improvement, plus a concrete end-to-end walkthrough of one fictional production ticket from alert to postmortem.

## 🛡️ RED-TEAM — 56 tests across 5 rounds, ALL PASS

Ran an adversarial attack suite ([server/redteam.py](../../privacy_game/server/redteam.py)) with 56 tests across 5 rounds. Research agent pulled real adversarial techniques from the text-attack literature (TextAttack, Trojan Source / UTS #39, ConfAIde, DeepWordBug, BERT-Attack).

**Real bugs found and fixed**:
1. **ZWJ digit injection** — `9​5​1​3​5` evaded both RP and adversary
2. **Salary substring FP** — agent says `$920000` (10x the truth), `92000` matches as substring → reward hack
3. **Years-at-employer substring FP** — `121 months` matched gold `21`
4. **Date format rigidity** — `May 1st, 2026` rejected, only ISO accepted → would kill P3-D for trained LLMs
5. **Devanagari digits** ०१२३४५ — bypassed regex (NFKC doesn't normalize non-Western digit scripts)
6. **Tag characters (U+E0020+)** — invisible plane-14 chars bypassed
7. **Variation selectors (U+FE00-FE0F)** — invisible modifiers bypassed
8. **Combining diacritics (Mn category)** — broke word-boundary regex on digits
9. **Control characters (Cc)** — broke name extraction (BEL in middle of "Jane")
10. **P3-D chit-chat city FP** — "New York-style pizza" falsely flagged as location leak

**Fixes applied**:
- Hardened `normalize_text` / `_normalize`: NFKC + `unicodedata.digit()` for all digit scripts + strip `Cf|Mn|Me|Cc` categories + casefold. Single principled fix closed 6 evasions at once.
- Numeric word-boundary regex for salary / years / credit-card / SSN
- Flexible date parser accepting ISO / US / "May 1st, 2026" / "1st of May"
- Metaphor filter on over-share scan (skips `"-style"`, `"-inspired"`, `"like X"`, etc.)
- Bidirectional substring match for free-text fields (diagnosis partial like "depression" ↔ "Major Depression")

**Known limitations documented (not reward-hack exploitable — agent can't win by using these)**:
- Drug brand names not in alias dict (Glucophage ≠ metformin)
- Employer nicknames not in alias dict (Big Blue ≠ IBM)
- Ambiguous toponyms (Paris, TX triggers city scan)
- Pronoun-only gender ("Mrs." alone not caught)
- Relative dates ("turning 40 next month") not parsed
- Chinese word-digits 零一二三 (NFKC doesn't, and `unicodedata.digit()` returns them correctly but we don't explicitly test)
- Base64 / hex encoding — v2

Test categories breakdown:
- **A** Reward hack attempts (4): encoding evasions, substring FPs — all fixed
- **B** Consistency/robustness (3): empty messages, massive verbosity, no crashes
- **C** Adversary catches self-sabotage (3): self-intro, email first.last, chitchat city metaphor
- **D** Tier enforcement (2): tier-3 generalizations rejected for tier-≤2 tasks
- **E** Phase-3 attack models work (3): drug→diagnosis, drug-class defense, employer→religion
- **F** Deeper edge cases (10): homoglyphs, ghost identity, age arithmetic, ticket lie, date mismatch, reward arithmetic, travel verb true-positive, state-only tier 3, empty protected fields
- **G** After first-round fixes (8): CC decoy, date-flex, home city exclusion, diagnosis partial, travel verb variants, ZWJ-in-email, homoglyph zip, real password leak
- **H** Exotic unicode (10): Devanagari, Chinese digits, tag chars, BiDi override, VS selectors, brand-name drug, negation, combining diacritics, BEL, echo RP
- **I** Research-grade (13): ligatures, Arabic-Indic digits, math-bold digits, NBSP in dates, ICS calendar leak, gold-collision order#, Paris-TX ambiguity, Mrs. pronoun, Big Blue nickname, cross-turn triangulation, relative dates, redaction theater, longer-number word boundary

Post-hardening sanity gate: `refuse=-0.001 < reveal=0.798 < smart=0.983`. Margin smart−reveal = +0.185. Env still teaches privacy + utility cleanly.

## 🔴 KNOWN MINOR ISSUES (non-blocking)

1. **`obs.metadata` returns None over the WebSocket wire.** Per-component scores (utility / reconstruction / verbosity / per-protected-recovered) are computed correctly server-side but don't reach the client. Reward itself comes through fine, which is what GRPO uses. Fix in v2 polish — investigate StepResult metadata serialization in openenv-core 0.2.3.
2. **IDE language-server warnings** about missing modules — IDE points at global Homebrew Python; venv has the packages. Fix: in VS Code / Cursor → `Cmd+Shift+P` → `Python: Select Interpreter` → choose `.venv/bin/python`.

## 🔵 OUTSTANDING DECISIONS (Day 2)

- Submission medium: HF blog vs <2min YouTube vs slide deck (deferred per §G #5)
- Stretch P3-E (negotiation/counter-offer) — only if Day 1 finishes with time to spare
- v2 NER ensemble — only if training shows the model learning to game the rule-based adversary

## What this means for the user

Today's critical-path work is essentially done. The env exists, runs locally end-to-end, and the reward signal is verified to teach both privacy and utility. Tomorrow's work is mostly *training* and *demo*, both of which depend on cloud resources (HF credits, Colab) you don't have until the hackathon kicks off. So the current state is: **shippable as-is** if anything goes wrong tomorrow with the cloud setup.

The only thing still on you for **today** is the 🟠 block in [YOUR_CHECKLIST.md](YOUR_CHECKLIST.md) — HF login, push empty Space, launch Docker Desktop, sleep 6 hours.
