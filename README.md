# META_H — OpenEnv Hackathon Finals (India, April 2026)

This repo is the build for the Meta OpenEnv Hackathon finals — a multi-agent RL environment named the **Contextual-Integrity Disclosure Game** that trains LLMs in *context-aware information control under adversarial inference*.

**TL;DR**: small chatbot learns to share enough to get tasks done, but not enough that an off-screen adversary can reconstruct who it is. Anchored to real privacy research (Sweeney 2000 quasi-identifiers, ConfAIde 2023 contextual integrity).

## Repo layout

```
META_H/
├── README.md                       (this file)
├── docs/                           context-files for builders + judges
│   ├── [4 hackathon docs from Meta]
│   └── notes/
│       ├── STRATEGIC_BRIEF.md      one-page strategy doc
│       ├── ENV_SPEC.md             v1 env contract (locked)
│       ├── DESIGN_DEPTH.md         edge cases, alternatives, phased plan
│       ├── EXPLAINED.md            kid-level + technical explainer
│       ├── PRIVACY_FRAME.md        4-skill taxonomy + MI framing for writeup
│       ├── PIPELINE.md             end-to-end real-world system view
│       ├── YOUR_CHECKLIST.md       user-facing action list
│       └── STATUS.md               live progress tracker
├── privacy_game/                   the OpenEnv environment package
│   ├── README.md                   env-specific README (HF Space card)
│   ├── client.py / models.py
│   ├── server/                     env logic, RP, adversary, profiles, tasks, baselines, redteam
│   ├── voice/                      voice layer (TTS, ASR, demo render, cross-modal eval, voice redteam)
│   └── notebooks/grpo_train.py     Colab GRPO training scaffold
└── .venv/                          Python 3.13 virtualenv (created by setup)
```

## Where to start

- **Want to understand the project?** Read [docs/notes/EXPLAINED.md](docs/notes/EXPLAINED.md) — kid-level intro with 25+ examples.
- **Want the strategy?** Read [docs/notes/STRATEGIC_BRIEF.md](docs/notes/STRATEGIC_BRIEF.md).
- **Want the formal framing for the writeup?** Read [docs/notes/PRIVACY_FRAME.md](docs/notes/PRIVACY_FRAME.md) — 4-skill taxonomy + MI lens.
- **Want the env technical spec?** Read [docs/notes/ENV_SPEC.md](docs/notes/ENV_SPEC.md).
- **Want to run the env?** See [privacy_game/README.md](privacy_game/README.md).
- **Want to know what's done and what's next?** [docs/notes/STATUS.md](docs/notes/STATUS.md).

## Run the env locally

```bash
source .venv/bin/activate
cd privacy_game
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Browse http://localhost:8000/web for the playable web UI
```

## Run sanity baselines

```bash
cd /Users/rajveerbishnoi/META_H && source .venv/bin/activate
cd privacy_game
python -m server.baselines --n-episodes 200 --n-profiles 100
# Should print: ✅ PASS — env reward signal teaches privacy AND utility.
```

## Hackathon submission targets

- ✅ OpenEnv-compliant environment (`openenv.yaml`, FastAPI server, Docker build, valid client)
- ✅ Red-team hardened (56 text tests + 7 voice tests; unicode evasions / substring FPs / reward hacks all closed)
- ✅ Voice extension (TTS + Whisper ASR + cross-modality eval + demo render)
- ✅ Docker image builds locally, container serves `/health`
- ✅ End-to-end smoke test: Python SDK → WebSocket → reward=1.000
- ⏳ Hugging Face Space hosting the env (push tomorrow, needs `huggingface-cli login`)
- ⏳ TRL GRPO training run on Colab
- ⏳ Reward curves + Pareto plot + held-out eval plots committed
- ⏳ <2 min YouTube demo — audio assets rendered, needs video editing
- ✅ Citation list ready (Sweeney 2000, ConfAIde 2023, Nissenbaum 2010, GRPO/DeepSeek-Math, OpenEnv)

## Voice extension — key research finding

From `voice/audio_eval.py` (cross-modality test): **the text-trained `smart` policy's privacy transfers cleanly to voice** (Δrecon = 0 across all four P3 tasks). The `reveal` policy leaks LESS in voice than in oracle text because Whisper `base.en` mis-transcribes rare medical entities (metformin → "met for men", efavirenz → "a faverens") ~60% of the time. Honest framing: voice modality offers accidental privacy for naive policies; the trained agent earns its privacy regardless.
