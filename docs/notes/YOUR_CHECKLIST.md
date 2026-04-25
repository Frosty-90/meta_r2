# YOUR CHECKLIST — things only you can do, with timing

Your personal action list. I can't do these for you. Tick them as you go.

---

## 🔴 RIGHT NOW (next 15-30 min) — these unblock me

**These 6 must happen before I can start coding the env.**

- [ ] **Read [DESIGN_DEPTH.md §G](DESIGN_DEPTH.md)** (the 5 sign-off questions at the bottom).
- [ ] **Answer §G question 1**: keep all four Phase-3 tasks (P3-A k-anonymity, P3-B drug→diagnosis, P3-C employer leak, P3-D over-sharing)? Drop any? Add any?
- [ ] **Answer §G question 2**: tolerant-extractor calibration — `zip-3` worth 0.8 utility, `state-only` worth 0.5, `year-only DOB` worth 0.8, `decade DOB` worth 0.5, `drug class` worth 0.7. Yes/no?
- [ ] **Answer §G question 3**: Phase 1 / Phase 2 — write task templates as concretely as Phase 3, or trust them as "easy mode" sketches?
- [ ] **Answer §G question 4**: stretch goal at Day-2 hour 6 — **GPT-4-as-adversary headline** vs **frozen-LLM Relying Party**? (My vote: GPT-4 adversary.)
- [ ] **Answer §G question 5**: submission medium — HF blog / <2min YouTube / slide deck? (My vote: <2min YouTube.)

---

## 🟠 TODAY (research/build day, 2026-04-25) — environmental setup

**Do these in parallel while I'm coding the env. They're cheap individually but lethal if missed.**

### Account & access (~30 min total, do early)

- [ ] **HuggingFace account ready**: log into [huggingface.co](https://huggingface.co), confirm you can see your profile.
- [ ] **HF CLI logged in on your Mac**: open Terminal, run `huggingface-cli login`, paste a write-scoped token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens). Verify with `huggingface-cli whoami`.
- [ ] **Push an empty test Space NOW** to verify push flow works on your account — pick any blank template at [hf.co/new-space](https://huggingface.co/new-space). If push fails today, you'll be debugging it tomorrow at 3am instead.
- [ ] **Google account for Colab**: log into [colab.research.google.com](https://colab.research.google.com), confirm runtime → "Change runtime type" → GPU is offered.
- [ ] **GitHub account ready** (for cloning OpenEnv repo + pushing your own backup): `gh auth status` or just `git clone <test repo>` to verify SSH/HTTPS auth works.

### Local toolchain (~30 min)

- [ ] **Python 3.10+** installed: `python3 --version` → 3.10 or higher.
- [ ] **pip works**: `pip --version`.
- [ ] **Docker Desktop installed and running** on the Mac (we need this for `openenv` Docker build): [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop). Open the app, confirm whale icon in menubar is steady.
- [ ] **At least 10 GB free disk space**: `df -h ~ | head -2`. Docker images + datasets + models add up fast.
- [ ] **Install `openenv` CLI**: `pip install openenv-core`. Verify `openenv --version`.
- [ ] **Install backup tools**: `pip install presidio_analyzer faker datasets`. (You don't need to install training libs locally — those go to Colab tomorrow.)

### Project decisions (~15 min, only you can decide)

- [ ] **Pick a team name** for the HF Space (e.g., `your-username/privacy-game-env`). Tell me what it is so I can use the right name in code.
- [ ] **Decide who's the "demo presenter"** on your team. They'll do the voiceover for the video tomorrow.
- [ ] **Confirm HF compute story**: does your hackathon ticket include HF Pro, HF Pro+, or just credits? When does it activate? (Affects whether we get T4, A100, or H100 tomorrow.)

### Team coordination (~30 min, your call entirely)

- [ ] Decide your team's sub-roles for tomorrow (env code, dataset, training, demo, README) — you said you'd manage this, so I'll trust the split.
- [ ] **Share this `docs/notes/` folder with teammates**. Walk them through [EXPLAINED.md](EXPLAINED.md) so everyone's mental model is identical.
- [ ] **Confirm where you're meeting tomorrow** (venue address, time, wifi situation).

### Pre-sleep hygiene (~15 min)

- [ ] **Charge laptop, phone, headphones, charger.**
- [ ] **Phone hotspot ready** in case venue wifi is weak.
- [ ] **Restart your Mac** before bed — clears memory, kills lingering processes.
- [ ] **Set 2 alarms** for tomorrow.
- [ ] **Sleep at least 6 hours.** Not negotiable. Tired you on day 2 = bad code + missed kill-switch decisions.

---

## 🟡 TOMORROW MORNING (2026-04-26, before hackathon kickoff)

### Pre-departure (~15 min)

- [ ] **Bring**: laptop, charger, headphones, water bottle, snacks, phone, hotspot cable.
- [ ] **Wear something comfortable**. You'll be sitting 10+ hours.
- [ ] **Eat a real breakfast.** Coffee alone is not breakfast.

### At the venue, before kickoff (~15 min)

- [ ] **Connect to venue wifi**. If it's bad, switch to phone hotspot.
- [ ] **Re-verify all logins** (HF, Colab, GitHub) — sometimes tokens expire overnight.
- [ ] **Pull the latest of our repo**: `git pull` in `/Users/rajveerbishnoi/META_H` (or wherever the env repo lives).
- [ ] **Open Colab**, create a new notebook, name it `privacy_game_grpo_train.ipynb`.

---

## 🟢 TOMORROW DURING HACKATHON

### Hour 0 (kickoff)

- [ ] **Listen to the kickoff for HF credits info.** Activate immediately.
- [ ] **In Colab**: switch runtime to A100 if available, else T4. Run `!nvidia-smi` to confirm.
- [ ] **Tell me the moment HF credits are live** so I can adjust the model size in the training notebook.

### Hour 1-3 (training begins)

- [ ] **Start the training run** (I'll have the notebook ready). Don't watch the loss bar — go work on README/demo prep instead.
- [ ] **Set a 30-min timer** — every 30 min, sample 5 rollouts and skim them for cheating.

### Hour 3-6 (mid-training)

- [ ] **Lunch.** Don't skip. (Seriously.)
- [ ] **At hour 3**: check reward curve. If flat, we drop to Phase 1+2 only. Tell me, I'll patch the curriculum.
- [ ] **At hour 6**: check whether trained model shows strategic disclosure on Phase 3. If yes, we push for the full demo. If no, we trim claims.

### Hour 6-9 (eval + demo)

- [ ] **Held-out eval runs.** I'll have the script — you just hit run.
- [ ] **Record the demo video** (you or your demo-presenter). 3 scenes per [DESIGN_DEPTH.md §E](DESIGN_DEPTH.md). Use OBS, QuickTime, or Loom. Aim for <2 min. Don't try to be perfect — V1 captured > V2 imagined.
- [ ] **Write the HF blog post** (or upload the video to YouTube).

### Hour 9-10 (kill-switch + polish)

- [ ] **HOUR 9 KILL-SWITCH DECISION**: is the trained model meaningfully better than baseline?
  - **YES** → push the strong story. We win on innovation + storytelling.
  - **NO** → fall back to the cherry-picked qualitative wins. Don't claim aggregate improvement we don't have.
- [ ] **README final polish**: every section has content, every plot embedded, every external link opens.

### Hour 10-12 (submit + breathe)

- [ ] **SUBMIT.** Whatever the hackathon's intake form is. **Do this with at least 1 hour of buffer before deadline.** Things go wrong at the last minute.
- [ ] **Verify submission accepted** — refresh, re-read confirmation email.
- [ ] **Tell teammates we're done.** Eat. Sit down. Watch other teams' demos. You earned it.

---

## 🟣 THINGS TO KEEP IN YOUR HEAD (memorize these)

- **The 1-paragraph elevator pitch** — see [EXPLAINED.md §14](EXPLAINED.md). Re-read it tonight, say it out loud.
- **The "Sweeney 2000" citation** — `{ZIP-5, full DOB, gender}` uniquely identifies 87% of US residents. Drop this on any judge who asks "why is this novel?"
- **The killer demo line** — "Same model, same input. Before training, the adversary recovers the user's name with 0.94 confidence. After training, with 0.18. We didn't change the model's knowledge. We changed its instinct."
- **The "why RL not SFT" answer** — "There is no fixed gold response in our env. The right disclosure depends on what was already said. SFT can teach 'reveal X' or 'refuse X'; it cannot teach 'reveal *part* of X conditioned on what's already in the transcript.' That cumulative reasoning emerges only from rollout reward. We can prove it: SFT baseline plateaus where RL keeps climbing."

---

## 📋 FINAL CHECK (5 min before submit)

- [ ] README has motivation paragraph, env description, training command, results plots.
- [ ] All plots are committed to repo (not just in Colab).
- [ ] HF Space is public and runs.
- [ ] Demo video link works in a private/incognito browser.
- [ ] HF blog post is published (if you went that route).
- [ ] Pinned library versions documented in README + Colab.
- [ ] Citation list includes Sweeney 2000 + ConfAIde / Mireshghallah 2023 + GRPO / DeepSeek-Math.
- [ ] Submission form filled with the right URL.

---

## 🆘 IF SOMETHING IS ON FIRE

- **Training won't start** → tell me immediately, I'll switch to smaller model + scripted-policy fallback.
- **HF Space won't deploy** → train against in-Colab Docker container; submit Space later.
- **Colab disconnects** → use Colab Pro auto-reconnect; or save adapter checkpoints every 50 steps so we can resume.
- **Reward curve flat** → drop Phase 3, train only Phase 1+2, ship that as the core result.
- **Demo video too long** → cut scene 3 (reward curve), put curve in README only.
- **Submission deadline approaching with bugs** → submit what you have. A 70% submission counts. A 0% submission doesn't.

---

You handle the human stuff. I handle the code. Let's win this.
