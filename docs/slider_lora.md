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
- **2026-06-16** — **S2 done** (branch `feat/slider-config` off `feat/slider-core`):
  `TrainingMethod.SLIDER`; `SliderRegime` enum (PROMPT_PAIR / IMAGE_PAIR);
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
  - **Deferred:** image-pair regime (raises NotImplementedError for now); multiplier-
    sweep sampling (samples currently show the adapter at the last-set multiplier,
    ≈ +strength); high-σ timestep weighting (currently uniform in [sigma_min,max]).
- **2026-06-16** — **S5 image-pair done** (branch `feat/slider-image` off
  `feat/slider-anima`): the visual (CS Eq. 9) regime is now runnable on Anima.
  - Config: `SliderImagePairConfig` (before/after/prompt/weight triple) +
    `TrainConfig.slider_image_pairs` (round-trips via the same BaseConfig list path
    as `slider_prompts`).
  - `AnimaSliderSetup`: `predict` now dispatches by regime — `_predict_prompt_pair`
    (unchanged) and the new `_predict_image_pair`. The image-pair step picks a
    weighted pair, VAE-encodes before/after to scaled latents (cached per path via
    `_encode_image`, mirroring AnimaBaseDataLoader: RGB→[-1,1]→5D→`vae.encode().
    latent_dist.mean`→`scale_latents`), builds the rectified-flow target at a
    sampled σ (`_make_flow_target`: `x_t=(1-σ)x0+σ·noise`, `v=noise-x0`), and runs
    the existing `_slider_image_pair_loss` (−strength⇒before/A, +strength⇒after/B)
    under the pair's conditioning. **No frozen-base forward and no eta** — the flow
    target IS the supervision. The datasetless loader is reused as-is (it just
    drives the step count; the setup owns the images).
  - UI: a "Image pairs · edit image pairs…" button on the Slider tab opens a popup
    (`SliderImagePairWindow`) hosting a before/after `ConfigList` with image
    `path_entry`s — kept out of the main tab so it doesn't crowd the prompt-triple
    list the prompt-pair user is testing.
  - Tests (CPU, `tests/test_anima_slider.py`): weighted pair selection,
    `_make_flow_target` rectified-flow identity, `_encode_image` preprocessing
    (cover-crop to the configured resolution → 5D → /8 latent) + per-path caching,
    and the empty-pairs guard. GPU-validated forward path still pending.
  - **`prompt` field = the combined-regime hook** (see §9): empty ⇒ pure
    empty-prompt image-pair slider; non-empty ⇒ the visual A/B target is learned in
    that prompt's context (a prompt-anchored image example).
- **2026-06-16** — **Image-slider design pivot (design discussion, §10).**
  The explicit-pair image path above (`slider_image_pairs` list + datasetless
  self-encoding) is **superseded as design-of-record** by coordinate-labeled image
  sliders (§10): vanilla concepts + declared-axis caption tokens `(axis:value)`,
  routed through the real MGDS pipeline. Rationale chain (derived in discussion,
  not yet built): pairs aren't load-bearing — the only real constraint is *the
  systematic pole difference must be the attribute and nothing else*, satisfied by
  matched pairs (cancel confounds per-example) **or** balanced shuffled sets
  (average them out). "0 = base" is accepted as the neutral, which selects the
  *reconstruction* objective (no A↔B coupling) over *transport* (where pairs would
  be in-gradient but 0≠neutral). Per-sample captions risk leaking the axis into the
  prompt (entangled control); the fix is attribute-⊥ conditioning, achieved cleanly
  by carrying the axis as an extracted caption coordinate and leaving the rest as
  context. The explicit-pair code on `feat/slider-image` is now a throwaway
  prototype; keepers = the mixin loss (unchanged), the regime split, and
  `_make_flow_target`/`_encode_image` plumbing (reusable). Prompt-pair
  (`feat/slider-anima`) is unaffected.

---

## 9. Combined / hybrid regime (forward-looking — not yet built)

Prompt-pair and image-pair are the two CS objectives; a *combined* regime fuses
them so the slider gets both a **semantic direction** (from the frozen base's
c+/c− guidance) and **visual grounding** (from concrete before/after exemplars).
The pieces are already in place to add it as a third `SliderRegime.COMBINED`:

- **Hooks that exist now:** the mixin exposes *both* losses
  (`_slider_prompt_loss`, `_slider_image_pair_loss`); the setup has both predict
  paths and both caches; the config carries both lists; image pairs already carry
  an optional `prompt`.
- **Design A — prompt-anchored image pairs (cheapest, partially live):** set a
  `prompt` on each image pair. The visual A/B reconstruction is then learned in
  that prompt's context, improving locality/disentanglement without a second
  objective. Already runnable today via the `prompt` field.
- **Design B — joint loss:** per step, sum `λ·prompt_pair_loss +
  (1-λ)·image_pair_loss` on a shared adapter (and ideally a shared (x_t, σ)
  schedule). The prompt pair sets the direction; the image pair pins it to real
  exemplars. Needs a `slider_combined_lambda` field and a `_predict_combined` that
  calls both bodies and blends — small once the σ sampling is factored out (a
  `_sample_sigma(config, rand)` helper both paths can share).
- **Design C — visual guidance direction:** replace c+/c− text with a *latent*
  direction `v(x_after) − v(x_before)` under empty prompt as the guidance delta fed
  into the prompt-style target `v(c_t)+η·Δ`. Lets a slider be defined purely by
  example images yet still trained with the guided-diff (not just reconstruction)
  objective — useful when the attribute is hard to phrase but easy to show.
- **Open question:** whether B/C beat plain prompt-pair enough to justify the
  per-step cost (B roughly doubles forwards). Decide empirically after the
  prompt-pair and image-pair baselines are GPU-validated. Until then the `prompt`
  field (Design A) is the no-cost entry point.

---

## 10. Coordinate-labeled image sliders (design of record — supersedes explicit pairs)

The image-slider data model. Replaces the explicit `slider_image_pairs` prototype.
Derived from first principles in design discussion (see §10.0); not yet built.

### 10.0 What actually constrains slider data (the reasoning)

- **One load-bearing constraint:** the adapter learns whatever *systematically
  differs between the two poles*, so the systematic pole difference must be the
  target attribute and **nothing else**. Confounds must be either **shared** (so
  they cancel) or **balanced/averaged** (so they integrate out).
- **Fixed pairs are NOT required.** They're one way to satisfy the constraint
  (cancel confounds per-example, data-efficient). Balanced, shuffled pole-sets are
  an equally valid way (average confounds out, data-hungry). The mixin's image-pair
  loss has **zero A↔B coupling** (two independent reconstructions), so co-occurrence
  of a "pair" in one step does nothing in-gradient — its only value is set balance.
- **"0 = base" (accepted).** The base model at multiplier 0 is the neutral. This
  selects the **reconstruction** objective (−s→pole−, +s→pole+, independent), for
  which 0=base is native — over a **transport** objective (noise around A, target
  shifts A→B) where pairs *would* be in-gradient but 0 would equal a pole or a
  blurry midpoint. Transport is parked behind the 0=base decision (a possible
  future objective, opt-in, knowingly trades away base-as-neutral).
- **Caption ⊥ polarity is mandatory.** If the conditioning text correlates with the
  axis (esp. naming the attribute — "close up" vs "far away"), the base reads the
  attribute from the prompt and the adapter only learns the residual ⇒ entangled,
  weak control whose effect depends on the inference prompt. So the axis must NOT
  live in the conditioning text.

### 10.1 The model

Vanilla OT **concepts**, unchanged (folders + sidecar captions + full MGDS
pipeline: aspect buckets, latent caching, augmentation). The slider axis lives in
the **caption**, per image, as a declared-axis coordinate token:

```
caption sidecar:   1girl, red dress, (distance:-2)
                   1boy, (distance:+1), (age:+1)
slider config:     declares axis names = ["distance", ...]
```

A pre-tokenize pipeline step (one `MapData` node) splits each caption into
`(clean_caption, {axis: float})`: it extracts `(<declared-axis>:<number>)` for the
declared axes only — passing every other `(token:weight)` through verbatim, so
a1111 attention weights are untouched — and the **coordinate leaves the
conditioning** (clean_caption goes to the encoder, coordinate rides as batch
metadata). That extraction is what makes the conditioning attribute-⊥ for the
labeled axis automatically.

### 10.2 Objective

Coordinate-scaled reconstruction. For an image at coordinate `ℓ` on the trained
axis, train the adapter at multiplier `m = k·ℓ` to reconstruct it (flow target
`v=noise−x0` at a sampled σ, via `_make_flow_target`). Binary polarity is the
special case `ℓ∈{−1,+1}`; a continuum of `ℓ` teaches a **calibrated, monotonic**
response across the whole sweep (= our definition of slider quality). The learned
single low-rank direction `d` must satisfy `base + m·d ≈ reconstruct(image@ℓ)` for
all samples; off-axis variation is residual and averages out (low rank = the
disentangler). `ℓ=0` samples are inert for the adapter (delta scaled by 0) — free
"this is neutral" assertions, usable for eval, not direct training signal.

### 10.3 v1 scope vs fast-follows

**v1 (the build):**
- Vanilla concepts; declared-axis token extraction; clean_caption → encoder,
  coordinate → multiplier `m=k·ℓ`.
- Coordinate is a **float**, consumed roughly as-is (single global gain `k`).
  Ordinal use (a few levels) is **recommended in docs, not enforced** — so strong
  continuous labels (lidar distance, measured age) flow through unquantized.
- **Balancing:** default **coordinate-symmetric** sampling (equalize across the
  coordinate sign/range so asymmetric datasets don't bias the direction). Opt-in
  **stratify over a declared confounder axis** (hold a labeled confounder's
  distribution constant across the trained coordinate — the continuous Eq. 8
  preservation mean; this is the payoff of *labeling* confounders rather than
  curating matched folders).
- Image sliders route through a **real MGDS dataloader** (not the prompt-pair
  datasetless loader), tagging each item with its coordinate(s).

**Required dataset-prep discipline in v1** (each becomes an automated fast-follow):
- Keep attribute *prose* out of captions → fast-follow: a configured list of
  **axis-linked phrases** to auto-strip from the conditioning.
- Put axis positions on a sane scale (matching the multiplier band) →
  fast-follow: **per-axis label rescaling/normalization** (so analog units just
  work without manual scaling).

**Deferred (agreed valuable, later):**
- **Caption-cluster stratification** — infer confounder contexts from the stripped
  captions and balance within them, for users who won't explicitly label
  confounders.
- **Transport objective** (§10.0) for true in-gradient paired training.
- Auto-rescale, auto-strip (the two prep steps above).

### 10.4 What carries over from the explicit-pair prototype

Keep: `_slider_image_pair_loss` (mixin, unchanged — coordinate scaling just sets
the multiplier it's already given), the `predict` regime split,
`_make_flow_target`, and `_encode_image` (VAE encode → scaled latent). Drop:
`SliderImagePairConfig` / `slider_image_pairs` / the popup pair editor / the
datasetless loader for the image path.
