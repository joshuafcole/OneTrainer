# Slider LoRA training in OneTrainer — design & working doc

> Living reference for adding **slider-LoRA (Concept Sliders)** training to OneTrainer.
> Initial target: **Anima** (flow-matching Cosmos-Predict2 DiT). Status log at the bottom.

---

## 0. Goal & scope

Add high-quality slider-LoRA training to OneTrainer. A *slider* is a LoRA whose
**signed multiplier** is a continuous control knob: `+α` pushes a concept one way,
`−α` the other, `0` is the base model. Both methodologies are in scope:

- **Prompt-pair sliders** (Concept Sliders, text-only, no image dataset)
- **Image-pair / anchor sliders** (Concept Sliders Eq. 9, 3–6 before/after pairs)
- **Novel hybrid:** gradient-aligned (GA) init of the slider from the
  guidance-difference gradient — leapfrog training. See §6.

PEFT network: default **low-rank standard LoRA**; LoKr supported but considered
overkill for sliders. GA-init (currently LoKr-only) to be **backported to plain
LoRA** so sliders can leapfrog regardless of network type.

---

## 1. The method (verified research)

Sources: Concept Sliders (Gandikota et al., ECCV 2024, arXiv 2311.12092),
project page sliders.baulab.info, official repo github.com/rohitgandikota/sliders,
ostris ai-toolkit. Claims below were adversarially verified (24/25 confirmed).

### 1.1 Core objective (ε-prediction space, Eq. 7)

A LoRA is trained so the **adapter-applied** prediction equals the **frozen base**
prediction shifted along a guidance direction the base itself supplies:

```
εθ*(x_t, c_t, t)  ←  εθ(x_t, c_t, t)  +  η·( εθ(x_t, c+, t) − εθ(x_t, c−, t) )
```

- `εθ` is **frozen**; only LoRA `θ*` trains. Base supplies `(c+ − c−)`; the LoRA
  learns to reproduce that disentangled direction at positive multiplier.
- Derived from a Bayesian score-shift (Eq. 5–6) via Tweedie's formula.
- `η` = **training-time** guidance scale (paper examples: 3–4).

### 1.2 Disentanglement (Eq. 8)

Sum the guidance difference over a **preservation set P** of protected attributes
(e.g. iterate race/gender prompts while editing age) so the learned direction
stays invariant to them.

### 1.3 Image-pair / visual sliders (Eq. 9)

No text. Train the LoRA in **both** directions on a before/after pair with an
empty prompt — negative-scaled LoRA reconstructs image A, positive-scaled
reconstructs image B. Works with **3–6 pairs**.

### 1.4 Inference strength is decoupled from training η

`W = W₀ + α·ΔW`. Scaling **α at inference** strengthens the edit without
retraining. ⇒ **the user-facing slider knob is the adapter multiplier, set
independently of η.** This is exactly what S1 (the runtime multiplier) provides.

### 1.5 Practical recipe (ostris ai-toolkit, verified)

- **Adapt the transformer only, not the text encoder** ("adjust the concept's
  representation, not its description"). Matches `AnimaLoRASetup` (TE frozen).
- **Polar-opposite prompt pairs.**
- **Standard LoRA, rank 8, alpha ≈ 4**, with an explicit warning that **higher
  rank is worse for sliders**. Low rank *is* the disentanglement mechanism.

---

## 2. Velocity-space adaptation (derived — the crux for Anima)

The literature has **no validated trained-slider recipe for flow-matching DiTs**
(official FLUX support is experimental/underperforms; FluxSpace [2412.09611] and
FlowSlider [2604.02088] are *training-free* inference-time editing, not trained
LoRAs). So the objective had to be derived for Anima's velocity target.

Anima is rectified flow: forward `x_t = (1−σ)x₀ + σε`, target velocity
`v = ε − x₀` (the code's `flow = noise − image`, `σ = t/T`). At fixed `(x_t, σ)`:

```
x₀ = x_t − σ·v        ε = x_t + (1−σ)·v
```

So for two conditionings at the **same** `(x_t, σ)`:

```
ε(c+) − ε(c−) = (1−σ)·( v(c+) − v(c−) )      (offset x_t cancels)
```

Substituting the CS ε-target and converting back to velocity, **the `(1−σ)`
factor cancels entirely**:

```
v*(x_t, c_t, t)  =  v(c_t)  +  η·( v(c+) − v(c−) )      ← velocity slider objective
```

**Result: the objective is form-invariant under ε → v — a direct substitution,
with NO per-timestep reweighting needed.** Eq. 8 (disentanglement) and Eq. 9
(image-pair) port identically. This rules out, by derivation, the main risk of
moving to flow matching.

**Load-bearing assumption (must verify first — Experiment #1):** the base must
respond to conditioning (true CFG, not guidance-distilled), else `v(c+) − v(c−)`
is trivial. Anima's sampler does a real 2-pass CFG, strongly implying it is *not*
distilled — but confirm empirically before building the full objective.

---

## 3. Composition characterization (reasoned; thin in literature)

**Network type.**
- Default: **low-rank standard LoRA** (rank 4–8, alpha ≈ rank/2). Verified best.
- **LoKr**: supported, but overkill for a single low-dim semantic direction.
  The signed multiplier scales the additive delta-W, well-defined for LoRA & LoKr.
- **DoRA / OFT**: signed/negative multiplier is **ill-defined** (DoRA's
  magnitude/direction split; OFT's orthogonal transform). Not for sliders.
- **rsLoRA** (alpha/√rank): changes how the multiplier maps to strength; pick one
  convention and document it (the multiplier *is* the slider knob).

**Module targeting (Cosmos DiT, existing `LAYER_PRESETS`).** Start with
`attn-only` or `attn-mlp` at low rank. `cross-attn` for purely text-semantic
attributes; self-attn carries spatial/style. `full`/`blocks` → bleed & fidelity loss.

**Optimizers & dynamics (reasoned).** AdamW or Prodigy; **low LR, few steps
(hundreds–low thousands), tiny effective batch** (each step runs several
forwards). Collapse modes: too-large η or too-high rank → global image shift /
fidelity loss. Monitor by **sampling a multiplier sweep**, not loss curves.

**Evaluation.** CLIP attribute-score **monotonicity** across a multiplier sweep;
**disentanglement** = CLIP similarity of *unrelated* attributes stays flat;
LPIPS/identity preservation for image-pair; human eval. High quality = monotonic,
locally disentangled, base fidelity preserved at multiplier 0.

---

## 4. OT gap analysis (code-grounded)

| Capability | Status in OT | Needed |
|---|---|---|
| Velocity slider objective | ✗ `BaseAnimaSetup.predict` is single-forward `{predicted, target=noise−image}` | new multi-forward predict + loss |
| Frozen-base passes at c_t/c+/c− | ✗ | toggle adapter via multiplier=0 under no_grad |
| **Runtime signed multiplier** | ✗ forward hardcoded `orig + W·(alpha/rank)` | **S1 (this branch)** |
| Prompt-pair / preservation config | ✗ no SLIDER in `TrainingMethod` | new config + enum |
| Datasetless `x_t` (SDEdit) | ✗ `predict` requires `latent_image` | new data path |
| Image-pair path | ◑ image→latent pipeline exists | reuse + A/B pairing |
| c+/c− text encoding | ✓ `model.encode_text` reusable | reuse |
| Factory dispatch | ✓ `factory.get(BaseModelSetup, model_type, training_method)` | register setup |
| Multiplier-sweep sampling | ◑ `AnimaSampler` does CFG | add sweep |

Key files: `modules/modelSetup/BaseAnimaSetup.py` (predict :394, loss :626),
`modules/module/LoRAModule.py` (PeftBase :22, LoRA :588, LoKr :287, forwards),
`modules/util/create.py:74` (factory), `modules/util/enum/TrainingMethod.py`,
`modules/trainer/GenericTrainer.py:805` (GA estimation pass).

---

## 5. Stacked-branch plan

Stacked on `feat/ga-init`. Each branch = one reviewable PR.

- **S1 · `feat/peft-multiplier`** *(in progress)* — runtime signed multiplier on
  `PeftBase`/`LoRAModuleWrapper`; enables base passes (m=0), ± training passes,
  and the inference slider knob. Foundational, model-agnostic.
- **S2 · `feat/slider-config`** — `TrainingMethod.SLIDER` + `TrainConfig` fields
  (target/positive/unconditional/neutral prompts, preservation set P, η, direction
  mode, regime, x_t-gen settings) + UI.
- **S3 · `feat/slider-core`** — model-agnostic slider objective mixin: velocity
  guided-diff loss (Eq. 7/8), image-pair loss (Eq. 9), SDEdit `x_t` generator.
- **S4 · `feat/slider-anima`** — `AnimaSliderSetup` wiring, data paths, factory
  registration, saver/spec, multiplier-sweep sampling.
- **S5 · `feat/slider-eval`** *(optional)* — CLIP monotonicity + disentanglement.

GA-init backport (§6) slots in as a prerequisite enhancement consumed by S3/S4.

---

## 6. GA-init synergy (the leapfrog)

OT's Kron-GA init (`GenericTrainer` :805 estimation pass → mean `dL/dW` per layer →
`LoRAModuleWrapper.init_lokr_from_gradients` → Van Loan factors, keeping the
zero-factor zero so the adapter output stays exactly zero at init).

**Backport to plain LoRA:** add `LoRAModule.init_from_gradient(grad)` — SVD the
estimated `dL/dW`, set `lora_down` = top-r right singular vectors (norm-matched),
keep `lora_up = 0`. After one optimizer step, `BA ≈ −lr·G_r` (rank-r gradient
step), i.e. the LoRA-GA first-step property, while the init delta is exactly zero.

**Why this is special for sliders:** at init (adapter = 0) the slider loss gradient
`dL/dW` is `∝ −η·(v(c+) − v(c−))` backpropped — the **guidance-difference
direction itself**. So GA-init pre-orients the slider adapter along exactly the
direction it must learn. Leapfrog. Works for both LoRA and LoKr.

---

## 7. Open experiments (literature can't answer)

1. **Anima CFG response / not distilled** — `v(c+) − v(c−)` must be non-trivial.
   Highest risk, cheapest. Probe script: `scripts/util/probe_anima_cfg_response.py`.
2. Which conditioning path (Qwen3 vs frozen T5 adapter) carries c+/c−; best preset.
3. LoKr + Kron-GA vs low-rank LoRA (+ GA) for disentanglement & multiplier linearity.
4. η / rank / alpha / step-count envelope; verify α/η decoupling holds in velocity space.
5. Prompt-only SDEdit `x_t` vs image-pair on Anima — quality/cost tradeoff.

---

## 8. Status log

- **2026-06-16** — Doc created. Research complete (deep-research, verified).
  Codebase mapped. Branch `feat/peft-multiplier` cut off `feat/ga-init`.
- **2026-06-16** — **S1 implemented** (`modules/module/LoRAModule.py`): `PeftBase.multiplier`
  + `set_multiplier`; applied in LoRA/LoHa/LoKr additive forwards; DoRA/OFT/SVD
  raise on non-default multiplier; `LoRAModuleWrapper.set_multiplier`. Unit test
  `tests/test_peft_multiplier.py` (CPU). Both files py-compile; **test not yet
  run** (no torch in the authoring sandbox — run in the OT venv).
- **2026-06-16** — **Experiment #1 probe** written:
  `scripts/util/probe_anima_cfg_response.py`. Measures rel_guidance =
  ‖v(c+)−v(c−)‖/‖v(c_t)‖ over prompt triples × timesteps via OT's own
  encode_text + Cosmos forward. Run on GPU before building S3.
- Next: run both (test + probe) in venv; then GA-LoRA backport (§6) and S2/S3.
- **2026-06-16** — **GA-LoRA backport done** (branch `feat/lora-ga-init` off
  `feat/peft-multiplier`). `LoRAModule.init_from_gradient`/`init_from_factors`
  (SVD of dL/dW → lora_down = top-r right singular vectors, norm-matched;
  lora_up stays 0). `LoRAModuleWrapper.init_lokr_from_gradients`/`_from_factors`
  now dispatch by type (Kron-GA for LoKr 2-tuple, LoRA-GA for LoRA 1-tuple,
  DoRA skipped). `GenericTrainer` GA gate broadened to `peft_type ∈ {LOKR,
  LORA}`; cache digest keyed by `peft_type` (+ `lora_rank` for LoRA) to avoid
  cross-type collisions; user-facing logs say "GA" not "Kron-GA". Test
  `tests/test_lora_grad_init.py`.
  **Known limitation:** the trigger is the LoKr-named `lokr_init_mode=GRADIENT`
  flag; surfacing a GA toggle on the LoRA UI tab is a follow-up (will fold into
  the S2 slider config). For sliders, the estimated dL/dW *is* the
  guidance-difference direction, so this pre-orients the adapter (the leapfrog).
- **2026-06-16** — Tests pass in venv (`test_peft_multiplier`, `test_lora_grad_init`).
  **Probe v1 result:** mean rel_guid=0.018 on *off-manifold Gaussian* x_t with raw
  (un-amplified) differences — borderline by an arbitrary 0.02 cutoff, BUT
  `cos_align` consistently positive (mean ~0.4) and the signal scaled correctly
  with timestep → a real, coherent conditioning direction, not a distilled/blind
  model. Two confounds identified: off-manifold latents and no "does it use
  conditioning at all" baseline. **Probe v2** (default now): on-manifold x_t via a
  short flow-matching Euler denoise, an empty-prompt baseline
  `rel_cfg=‖v(c_t)−v(∅)‖/‖v(c_t)‖`, and stronger polar prompts. Verdict keys on
  rel_cfg (conditioning-blind?) + cos_align (coherent axis?). Awaiting v2 numbers
  before committing to S3.
- **2026-06-16** — **Probe v2 verdict: VIABLE (green).** On-manifold means:
  rel_guid=0.029, rel_cfg=0.024, cos_align=0.54, **ratio≈1.2 (often >1)** — the
  attribute axis is as wide as the whole conditioning gap. Not distilled; coherent
  ± axis. Signal concentrates at high σ → **slider training should weight
  higher-noise timesteps** (design note for S4). Modest magnitude ⇒ η≈3–4 + enough
  steps, per the CS α/η decoupling. Core assumption (§2) holds.
- **2026-06-16** — **S3-core done** (branch `feat/slider-core` off
  `feat/lora-ga-init`): `modules/modelSetup/mixin/ModelSetupSliderMixin.py` —
  model-agnostic velocity slider objective. `_slider_prompt_loss` runs the frozen
  base (multiplier 0, no_grad) to build the detached target `v(c_t)+η·mean_p(v(c+,p)
  −v(c−,p))`, then trains ± passes via `set_multiplier`; `_slider_image_pair_loss`
  for CS Eq. 9. Decoupled from any model via `run_velocity`/`set_multiplier`
  callables. Test `tests/test_slider_objective.py` — a toy adapter trained with the
  loss learns the guidance direction (cos>0.99, rel_err<0.1) and the −strength pole
  mirrors +strength. **Next:** S2 (config/enum + GA UI toggle) and S4 (AnimaSliderSetup
  wiring: build x_t, the run_velocity closure, factory reg, multiplier-sweep sampling).
