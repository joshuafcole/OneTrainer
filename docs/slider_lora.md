# Slider LoRA training in OneTrainer — design & working doc

> Living reference for **slider-LoRA (Concept Sliders)** training in OneTrainer.
> Target: **Anima** (flow-matching Cosmos-Predict2 DiT). Both regimes are
> implemented; see **Current status** below, the chronological **§8 status log**,
> and **§10 Deferred / future work** at the very bottom.

---

## Current status (as implemented)

Both training regimes are wired end-to-end for Anima and unit-tested; **first
end-to-end GPU validation is still pending** (the authoring sandbox has no
torch/mgds, so the multi-forward predict paths are verified only by py-compile +
pure-function tests).

- **Prompt-pair sliders** — text-only, no dataset (§1, §2). Implemented in
  `AnimaSliderSetup._predict_prompt_pair` + `ModelSetupSliderMixin._slider_prompt_loss`,
  datasetless `AnimaSliderDataLoader`.
- **Coordinate-labeled image sliders** — real dataset; each caption carries an
  axis coordinate token `(distance:-2)` that becomes the adapter multiplier
  `m = k·coordinate` (§9). Implemented in `_predict_coordinate` +
  `_slider_coordinate_loss`, `AnimaSliderImageDataLoader`. **Supersedes** the
  explicit before/after-pair prototype (abandoned on `feat/slider-image`).
- **GA-init leapfrog** — gradient-aligned init backported from LoKr to plain LoRA
  (§6) and auto-engaged for slider runs.
- **UI** — `SliderTab` (regime-aware), TopBar "Slider" method for Anima only,
  shared LoRA tab. See §8 for the plumbing audit.
- A separate web-UI configuration/handoff spec lives in
  `docs/slider_training_config_spec.md`.

Branch stack (origin = the fork): `feat/peft-multiplier` → `feat/lora-ga-init` →
`feat/slider-core` → `feat/slider-config` → `feat/slider-anima` →
**`feat/slider-coordinate`** (current tip). `feat/slider-image` is abandoned.

---

## 0. Goal & scope

A *slider* is a LoRA whose **signed multiplier** is a continuous control knob:
`+α` pushes a concept one way, `−α` the other, `0` is the base model. Both
methodologies are implemented:

- **Prompt-pair sliders** (Concept Sliders, text-only, no image dataset) — §1, §2.
- **Coordinate-labeled image sliders** (real dataset; per-image caption coordinate
  → adapter multiplier; CS Eq. 9 generalized) — §9.
- **Gradient-aligned (GA) init** of the slider from the guidance-difference
  gradient — leapfrog training — §6.

PEFT network: default **low-rank standard LoRA**; LoKr supported but overkill for
sliders. GA-init was **backported from LoKr to plain LoRA** so sliders leapfrog
regardless of network type.

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

> *Implementation note:* OT generalizes this to **coordinate-labeled** sliders
> (per-image caption coordinate → multiplier `m=k·ℓ`; binary poles `ℓ∈{−1,+1}`
> are the special case). The explicit-pair form is the conceptual ancestor; see
> §9 for what was actually built and why pairs were dropped.

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

> **Resolved.** This was the original gap analysis. Every row below is now
> implemented (S1–S4 + §9); the one remaining ◑ is multiplier-sweep sampling,
> tracked in §10. Kept for context on what the work entailed.

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

Stacked on `feat/ga-init`. Each branch = one reviewable PR. **S1–S4 + the
coordinate-image branch are done and pushed; S5 is not built.**

- **S1 · `feat/peft-multiplier`** ✅ — runtime signed multiplier on
  `PeftBase`/`LoRAModuleWrapper`; enables base passes (m=0), ± training passes,
  and the inference slider knob. Foundational, model-agnostic.
- **GA-init backport · `feat/lora-ga-init`** ✅ — §6; consumed by S3/S4.
- **S3 · `feat/slider-core`** ✅ — model-agnostic slider objective mixin: velocity
  guided-diff loss (Eq. 7/8), image-pair / coordinate loss, SDEdit `x_t` generator.
- **S2 · `feat/slider-config`** ✅ — `TrainingMethod.SLIDER` + `TrainConfig` fields
  (prompts, preservation set P, η, regime, x_t-gen settings) + `SliderTab` UI.
- **S4 · `feat/slider-anima`** ✅ — `AnimaSliderSetup` prompt-pair wiring, datasetless
  loader, factory + saver/loader registration.
- **Coordinate image · `feat/slider-coordinate`** ✅ (current tip) — §9.
  Superseded the explicit-pair `feat/slider-image` (abandoned).
- **S5 · `feat/slider-eval`** ☐ *(optional, not built)* — CLIP monotonicity +
  disentanglement eval callback. See §10.

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

**Resolved:**
1. ✅ **Anima CFG response / not distilled** — probe v2 verdict **VIABLE**
   (rel_guid≈0.029, rel_cfg≈0.024, cos≈0.54, ratio≈1.2; signal concentrates at
   high σ). The §2 core assumption holds. Script:
   `scripts/util/probe_anima_cfg_response.py`. (See §8.)

**Still open (need the first real training runs):**
2. Which conditioning path (Qwen3 vs frozen T5 adapter) carries c+/c−; best preset.
3. LoKr + Kron-GA vs low-rank LoRA (+ GA) for disentanglement & multiplier linearity.
4. η / rank / alpha / step-count envelope; verify α/η decoupling holds in velocity space.
5. Prompt-pair SDEdit `x_t` vs coordinate-image on Anima — quality/cost tradeoff.

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
- **2026-06-16** — **S2 done** (branch `feat/slider-config` off `feat/slider-core`):
  `TrainingMethod.SLIDER`; `SliderRegime` enum (PROMPT_PAIR / IMAGE_PAIR — the
  latter later renamed `IMAGE`, see §9.3);
  `SliderPromptConfig` (target/positive/negative/weight triple) + `TrainConfig`
  fields `slider_{regime,prompts,preservation_prompts,eta,strength,symmetric,
  steps_per_epoch,anchor_steps,sigma_min,sigma_max}` (round-trips via BaseConfig
  list handling, like additional_embeddings). UI: new `SliderTab` (prompt-pair list
  + scalar settings; preservation prompts pipe-delimited single-line) shown for
  SLIDER alongside the shared LoRA tab; TopBar exposes "Slider" for Anima only.
  **GA toggle resolved:** the gradient-aligned init controls (mode/steps/gain/offload,
  reusing the `lokr_init_*` fields the GA backport generalized) now appear on the LoRA
  tab for `peft_type==LORA`, not just LoKr. Decision (user): prompt-pair is the
  primary S4 regime. All files py-compile. **Next:** S4 — AnimaSliderSetup + minimal
  datasetless prompt-pair loader + factory wiring (setup/saver/loader for ANIMA×SLIDER).
- **2026-06-16** — **S4 done** (branch `feat/slider-anima` off `feat/slider-config`):
  prompt-pair Concept Sliders now runnable on Anima.
  - `AnimaSliderDataLoader` (datasetless): bypasses MGDS, emits
    `slider_steps_per_epoch` trivial step-driver batches carrying only
    `concept_type=[STANDARD]*batch_size` (keeps the trainer's prior-prediction
    paths inert). Implements just the BaseDataLoader surface the trainer touches.
  - `AnimaSliderSetup(AnimaLoRASetup, ModelSetupSliderMixin)`: reuses all LoRA
    adapter construction; overrides `predict` to (1) pick a weighted prompt
    triple, (2) encode c_t/c+/c− (cached per prompt — prompts are fixed all run,
    so Qwen3+conditioner run once each), (3) build preservation-augmented pairs
    (bare pair + each `|`-delimited context; mixin averages the delta = CS Eq. 8),
    (4) generate on-manifold `x_t` by Euler SDEdit under the target at adapter=0
    (anchor_steps; 0 ⇒ off-manifold Gaussian), (5) run `_slider_prompt_loss` via a
    `run_velocity` closure over the Cosmos forward + `wrapper.set_multiplier`.
    `predict` returns `{loss}`; `calculate_loss` unwraps it (the multi-forward
    objective doesn't fit the single-forward predict/loss split). `setup_train_device`
    keeps Qwen3+conditioner (+VAE for sampling) resident — no text/latent cache.
  - Wiring: registered setup + (reused) LoRA saver/loader for ANIMA×SLIDER; sampler
    works via the existing model-type fallback. Broadened two `==LORA` gates to
    include SLIDER: BaseAnimaSetup autocast lora-dtype, and **GenericTrainer's GA
    init gate** — so GA-init now pre-orients a slider adapter along the
    guidance-difference direction (the §6 leapfrog) automatically.
  - Test `tests/test_anima_slider.py` (CPU): loader contract + setup helpers
    (weighted selection, preservation-pair construction, resolution parsing,
    per-prompt encode caching). Heavier forward path validated on GPU.
  - *(At this point the image regime raised NotImplementedError; it was later
    implemented as coordinate-labeled sliders — §9. Remaining deferrals are
    consolidated in §10.)*

## 9. Coordinate-labeled image sliders (design of record + as-built)

The visual-slider regime is **coordinate-labeled**, not explicit before/after
pairs. (An explicit-pair prototype was built on `feat/slider-image` and is
**superseded** — see the reasoning below for why.)

### 9.0 The one load-bearing constraint, and why pairs are not it

Derived with the user: the *only* hard requirement for a slider is that the
systematic difference between the two poles is the target attribute **and nothing
else**. Confounds must be either *shared* (matched pairs cancel them per example)
or *balanced/averaged away* (shuffled symmetric sets). Fixed pairs are merely one
data-efficient way to achieve confound control — and in our flow target they buy
even less than usual: `_slider_image_pair_loss` is two *independent*
reconstructions, so co-occurrence of a pair in one step has **zero in-gradient
coupling**. Pairs only ever bought set-balance (+ antithetic shared-noise variance
reduction). Therefore the right primitive is the **per-image axis coordinate**, not
the pair.

A second constraint follows: the caption must be **orthogonal to the axis**. If the
caption names the attribute, the frozen base reads it from the prompt and the
adapter has nothing to learn → entangled/weak control. So the coordinate must be
*removed* from the conditioning, not just recorded.

"0 = whatever the base does" (user's call) selects the **reconstruction** objective
(native neutral at multiplier 0) over a transport objective (where pairs would be
in-gradient but 0 ≠ neutral — parked).

### 9.1 The model

- **Vanilla OT concepts**, unchanged. Each image's caption carries an a1111-style
  coordinate token for its position on the axis, e.g. `a car on a road, (distance:-2)`.
- A **declared axis** list (`TrainConfig.slider_axes`, `SliderAxisConfig`). v1 =
  exactly one enabled **target** axis (`is_target`); its coordinate drives the
  multiplier. Other declared axes are still stripped from the caption (keep known
  confounders out of the conditioning) and `stratify` flags them for the balanced
  sampler (a fast-follow; unused in v1).
- The dataloader (`AnimaSliderImageDataLoader`, subclass of `AnimaBaseDataLoader`)
  injects two MGDS `MapData` nodes right after caption selection and **before**
  tag-dropout/tokenize: one extracts the target coordinate into a `(1,)` float
  tensor `slider_coordinate`; one strips every declared-axis token from `prompt`
  (a1111 emphasis on non-axis tokens passes through). Parsing is a torch-free pure
  fn, `modules/util/slider_caption_util.parse_slider_coordinates`. `slider_coordinate`
  is threaded through the cache split + output names so it survives latent/text
  caching.

### 9.2 The objective — coordinate-scaled reconstruction

Per sample: multiplier `m = k · ℓ` (`k` = `gain_k`, `ℓ` = raw coordinate), built at
step time so `k` can be retuned without rebuilding the cache. The adapter at `m`
reconstructs that image's rectified-flow target `v = noise − x0` at a shared sampled
σ (`_make_flow_target`). `ModelSetupSliderMixin._slider_coordinate_loss` sets each
sample's multiplier and MSEs its reconstruction, mean over the batch. With ℓ
symmetric around 0 across the dataset this yields a calibrated, monotonic slider;
binary poles `ℓ∈{−1,+1}, k=strength` recover `_slider_image_pair_loss` exactly (it is
the parity special case). No frozen-base guidance, no η — the flow target *is* the
supervision.

### 9.3 As-built reality (three corrections to the original design note)

Grounding against `BaseAnimaSetup.predict` surfaced three things the design note had
wrong, now fixed in code:

1. **Scale at step time.** MGDS `latent_image` is the *unscaled* VAE mean;
   `_predict_coordinate` applies `model.scale_latents` itself (the prototype's
   `_encode_image` pre-scaling does **not** carry over — that helper is dead for §9).
2. **Conditioning from the batch.** Uses `model.encode_text(qwen_hidden_states=
   batch["text_encoder_hidden_state"], tokens_t5=…)` per sample — *not* a fixed
   `_encode_cached(string)`. `_cond_cache`/`_encode_cached` are prompt-pair-only.
3. **General per-sample-multiplier loss.** `_slider_image_pair_loss` hardcodes
   even→−strength/odd→+strength by index parity and cannot express `m=k·ℓ`; the new
   `_slider_coordinate_loss` takes explicit per-sample multipliers.

What genuinely carries over: `_make_flow_target`, the `predict` regime split, the
`run_velocity_for_sample` closure shape. The `SliderRegime` value was renamed
`IMAGE_PAIR → IMAGE`.

### 9.4 v1 scope and fast-follows

- **v1 dataset prep (manual):** keep attribute prose out of captions; pre-scale the
  axis coordinates (ordinal recommended, not enforced — continuous/lidar labels work
  as-is via `gain_k`). Each image needs the target coordinate; a missing one yields
  ℓ=0 ⇒ m=0 ⇒ adapter-off ⇒ a wasted (harmless) sample. Disable tag-dropout for
  slider datasets (extraction runs before it, but a dropout-eaten coordinate tag
  would be lost).
- **Decisions (user):** single target axis; defer the stratified sampler; build now.
- **Fast-follows:** consolidated in §10 (auto-strip phrase list, per-axis rescale,
  stratified sampler, etc.).

- **2026-06-16** — **§9 coordinate image sliders implemented** (branch
  `feat/slider-coordinate` off `feat/slider-anima`; the explicit-pair
  `feat/slider-image` branch is abandoned/superseded). `SliderAxisConfig` +
  `TrainConfig.slider_axes`; `slider_caption_util`; `AnimaSliderImageDataLoader`
  (+ regime dispatcher in `AnimaSliderDataLoader`); `_slider_coordinate_loss`;
  `AnimaSliderSetup._predict_coordinate` + `_make_flow_target`/`_sample_sigma`/
  `_resolve_target_axis`; `SliderRegime.IMAGE`; `SliderTab` axes popup. Tests:
  parser (strip/passthrough/case/multi-axis), `_resolve_target_axis`,
  `_make_flow_target`, `_slider_coordinate_loss` multiplier schedule. All
  py-compile; parser smoke-tested. **Unverified beyond py-compile:** the real
  MGDS coordinate pipeline and the per-sample forward (no torch/mgds in the
  authoring sandbox) — first GPU run validates both regimes.
- **2026-06-16** — **UI plumbing audit + regime-aware Slider tab.** Verified the
  full config→UI→trainer path for both regimes: TopBar exposes "Slider" for Anima
  only; TrainUI adds the Slider + shared LoRA tabs for SLIDER and restores them on
  config/preset load (training-method selector fires its callback on init); the
  always-present Concepts tab supplies the IMAGE dataset; the pre-train gate
  (`flush_and_validate_all`) is field-level and does not hard-require concepts, so
  the datasetless PROMPT_PAIR regime is not blocked. Made `SliderTab` regime-aware:
  the regime selector now shows only the relevant block (prompt-pair settings +
  triple list, or the coordinate-axes editor) and a per-regime hint, hiding the
  prompt-triple list in IMAGE mode. **Known limitation:** misconfiguration (no
  enabled prompt triple / not exactly one target axis) surfaces as a clear
  RuntimeError at train start rather than a pre-flight field error (their
  validation framework is per-widget; cross-field slider validation is a
  fast-follow). Multiplier-sweep sampling still deferred (samples use the adapter's
  last-set multiplier).

---

## 10. Deferred / future work

Nothing below is implemented. Ordered roughly by value.

**Validation now (first GPU runs).**
- End-to-end GPU validation of both regimes — the multi-forward predict paths are
  verified only by py-compile + pure-function unit tests. Confirms: prompt-pair
  guidance forward; coordinate MGDS pipeline (in-place caption strip ordering +
  `slider_coordinate` surviving the cache) + per-sample forward/scale shapes.
- The open experiments in §7 (#2–#5): conditioning path, network type, η/rank/step
  envelope, prompt-pair vs coordinate-image quality/cost.

**Training quality.**
- **High-σ timestep weighting.** The probe found the attribute signal concentrates
  at high σ; both regimes currently sample σ uniformly in `[sigma_min, sigma_max]`.
- **Confounder-stratified sampler** (coordinate regime). The `SliderAxisConfig.stratify`
  flag is the hook; v1 relies on a coordinate-balanced dataset prepared by hand.
  Includes a coordinate-symmetric sampling default.
- **Transport objective** (coordinate regime, parked): pairs become in-gradient but
  `0 ≠ neutral`; we chose reconstruction so `0 = base` (§9.0).

**Dataset-prep automation (coordinate regime).** v1 requires manual prep:
- Auto-strip an axis-linked phrase list from captions (so the user need not keep
  attribute prose out by hand).
- Per-axis coordinate auto-rescale (v1 consumes raw coordinates × `gain_k`).

**Sampling / preview.**
- **Multiplier-sweep sampling.** During-training samples render at the adapter's
  last-set multiplier, not a `−s…0…+s` sweep, so they are not a true slider preview.
  True previews require post-train inference at several multipliers.

**Evaluation (S5, optional).** CLIP attribute-score monotonicity across a multiplier
sweep + disentanglement (unrelated attributes stay flat) + base fidelity at `m=0`.

**Combined / hybrid regime.** Joint prompt + image objective on a shared adapter/σ
(per-pair `prompt` is the hook), and/or a visual guidance direction
`v(x_after) − v(x_before)` fed into a prompt-style target. Decide empirically after
the baselines are GPU-validated.

**UX / platform.**
- Pre-flight cross-field slider validation surfaced in the UI (today misconfig is a
  clear RuntimeError at train start).
- Slider support for base models beyond Anima.
