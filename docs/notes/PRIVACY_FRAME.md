# PRIVACY_FRAME — the formal lens our writeup uses

This doc is the canonical source for how we *frame* the project to judges, in the README, in the HF blog, and in the demo voiceover. Every external-facing surface should pull from here. Internal env design follows ENV_SPEC.md; this doc is the *story* layer.

---

## 1. The capability we claim our env trains

> **Context-aware information control under adversarial inference.**

One sentence. Memorize it. Use it verbatim on the README first line, on the demo voiceover, and on the elevator pitch.

Three things in that sentence are doing work:
- **"context-aware"** — the right disclosure depends on what's already been said and what the relying party actually needs
- **"information control"** — not refusal, not honesty; *control* over the granularity, abstraction, and inclusion of what's revealed
- **"under adversarial inference"** — the threat is not direct extraction but *what an adversary can infer* from the combined disclosure

---

## 2. The four learnable skills (this is the README headline)

| # | Task | Skill | What the model has to learn |
|---|------|-------|-----------------------------|
| P3-A | Insurance underwriting | **Granularity control** | Generalize the *level of precision* (zip-5 → zip-3, full DOB → year only) so quasi-identifier combinations don't uniquely identify |
| P3-B | Pharmacy verification | **Semantic abstraction** | Generalize the *meaning* of a value (metformin → "an oral antidiabetic") so deterministic mappings to protected attributes are weakened |
| P3-C | Apartment rental | **Conditional generalization** | Generalize *only when the value actually leaks*; don't over-abstract harmless inputs |
| P3-D | Vacation mail hold | **Omission discipline** | Disclose what's required, *and nothing more* — don't volunteer context |

These four are deliberately distinct failure modes of naïve LLM honesty. Each requires a different kind of reasoning. Together they cover the space of "context-aware information control."

**Why this taxonomy is the headline frame**: it tells a judge in 4 rows what the env teaches. They don't have to wade through reward formulas to understand the contribution.

---

## 3. The formal lens (for the writeup / blog)

The optimization the agent learns to perform:

```
minimize    I(disclosed_transcript ; protected_attributes)
subject to  utility(disclosed_transcript) ≥ task_threshold
```

Where `I(·;·)` is mutual information between the disclosed conversation and the protected attribute set, and `utility(·)` is the relying party's task-completion check.

Cite once, then move on. The formalism does two jobs:
1. Signals to research-lab judges that we know the privacy literature.
2. Disambiguates our framing from "PII redaction" or "content moderation" — those are extraction tasks; ours is an *information-flow* control task.

We do **not** compute `I(·;·)` in the env. Reward is the simpler proxy:

```
reward = utility − reconstruction − verbosity_penalty
```

`reconstruction` is the empirical estimator of mutual-information leak: if the adversary can reconstruct the protected attribute, that *is* mutual information leaking. Cleaner than computing entropy.

---

## 4. The two attack models we defend against

Any privacy task is defined by who the adversary is. We name ours explicitly:

### Attack model 1 — Linkage attack (covers P3-A)
Adversary holds a side-channel database (registry, public records). Quasi-identifier attacks per Sweeney 2000.

### Attack model 2 — Attribute inference attack (covers P3-B, P3-C)
Adversary uses prior knowledge of correlations (drug → diagnosis, employer → religion) to infer protected attributes from non-protected disclosures.

P3-D is interesting because it's **self-inflicted leakage** — the agent volunteers information that was never asked for. Same threat model as Attack 2 but the agent is the source of the proxy.

Naming the attack models in the writeup is a credibility move. It tells reviewers we built against a real adversary, not a vibe.

---

## 5. Why this is genuinely an RL problem (the writeup paragraph)

Three properties together make this require RL:

1. **No fixed gold response.** For P3-A, multiple generalization strategies all win — zip-3 + year-only-DOB + male, OR state-only + decade + male, OR many others. SFT can't pick "the right" answer because there isn't one.

2. **Cross-turn cumulative leakage.** The right disclosure in turn 3 depends on what's already in turns 1-2. Reasoning over the full transcript distribution requires policy gradients, not flashcards.

3. **Adversary-defined optimum.** The Pareto frontier shifts based on what the adversary can do. Training against the adversary is fundamentally a game, not a labeling task.

> "We trained Qwen2.5-1.5B with GRPO against this environment. SFT on the same task succeeds at Phase 1-2 but plateaus on Phase 3 because no fixed gold response exists for cumulative-leakage cases. The RL agent learns to *reason about adversarial inference*, not to recite a redaction rule."

That's the paragraph that goes in the writeup.

---

## 6. The Pareto-frontier evaluation plot

For each Phase-3 task, scatter-plot the `(utility, 1 − reconstruction)` point of:

- 🔵 random policy
- 🟠 always-reveal
- 🟢 always-class-disclose / always-generalize
- 🔴 always-refuse
- ⚫ smart hand-played upper bound
- ⭐ **trained model**

Goal: trained model sits **Pareto-dominant** — no baseline matches its (utility, privacy) pair. If it doesn't, we have to be honest in the writeup; if it does, this is the headline plot.

This plot is more research-credible than a single reward curve because it shows the trained model isn't just optimizing one axis at the cost of the other.

---

## 7. Demo language anchored to this frame

| Demo moment | What the voiceover says |
|---|---|
| Open | "Modern LLMs leak information under adversarial inference. We built an arena that trains the opposite instinct." |
| Scene 1 (base model leaks) | "Same input. The base model gives full ZIP, full DOB, gender. The adversary's registry returns Jane Doe with 0.94 confidence." |
| Scene 2 (trained model defends) | "Same input. After RL training, the model gives ZIP-3 and year-only DOB. Quasi-identifier collapsed. Adversary returns nothing." |
| Scene 3 (curve) | "Across 500 GRPO steps, the model learned four distinct skills: granularity control, semantic abstraction, conditional generalization, and omission discipline." |

Every demo phrase pulls from this doc. Consistency across surfaces is its own credibility signal.

---

## 8. What we explicitly *don't* claim

Stay honest. The writeup should not over-promise.

- We do **not** provide formal differential-privacy guarantees. Future work.
- We do **not** train a learned re-identification adversary. Rule-based + NER ensemble. Future work.
- We do **not** test against a frontier-model adversary in the main eval (stretch only).
- We do **not** generalize to unseen protected attribute *types*. The 24-field vocabulary is fixed.

Listing these in a "Limitations" section of the README is a research-credibility move — under-promising and over-delivering plays better than the inverse.

---

## 9. Citations to drop in the writeup

- **Sweeney 2000** — "Simple Demographics Often Identify People Uniquely." Anchors P3-A.
- **Mireshghallah et al. 2023, "Can LLMs Keep a Secret?" (ConfAIde)** — anchors the framing that LLMs over-share.
- **Nissenbaum 2010** — "Privacy in Context" / contextual integrity. Theoretical anchor for "the rules depend on context."
- **DeepSeek-Math (Shao et al. 2024)** — anchors the GRPO method.
- **Meta OpenEnv / TRL OpenEnv integration** — infrastructure citation.

Five citations is enough. More would look like padding.

---

## 10. The reduction-to-known-problems table

In the limitations / future work section, we honestly map our env to known privacy primitives:

| Our task | Closest formal primitive | What we approximate |
|---|---|---|
| P3-A | k-anonymity (Sweeney) | We measure re-identification empirically via registry lookup, not a formal k threshold |
| P3-B | Attribute inference / inferential privacy | We use deterministic rule-based mappings; future work could use learned classifiers |
| P3-C | Contextual integrity (Nissenbaum) | The relying party's *legitimate* needs vs. *inferential* leaks correspond to Nissenbaum's "norms of appropriateness" |
| P3-D | Minimum-information principle | Standard data-minimization principle (GDPR Art. 5(1)(c)) operationalized as a learned policy |

This table is gold for the writeup because it grounds every task in real research — not as toy reproductions but as *operationalizations* of real principles into trainable RL tasks.
