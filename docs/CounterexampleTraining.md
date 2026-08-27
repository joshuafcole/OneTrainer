# Counterexample Training

Train a concept **away from** close-but-wrong images, alongside the positives it
is trained toward.

Set a concept's **Type** to `COUNTEREXAMPLE` in the Concepts tab. Its images are
near-misses — renders from a previous run that a human marked bad — captioned
**with the same trigger as the positives**. A near-miss captioned *without* the
trigger is a different experiment (concept ablation via an anchor), not this one.

## What it does

For each counterexample row, at the same `(x_t, t)` the positive rows use:

```
d      = ‖ v_θ(x_t, c, t) − v ‖²     trained  (adapter on)
d_ref  = ‖ v_ref(x_t, c, t) − v ‖²   frozen   (adapter off)
Δ      = d_ref − d      # > 0  ⇔  the adapter fits the WRONG image
                        #          BETTER than the base model does
L      = (2/β) · softplus(β · Δ)
```

`v` is the batch's target: the velocity target on a rectified-flow model, the
noise target on an epsilon-prediction one. Both entry points route the term.

The row's ordinary reconstruction loss is **replaced** by `L`, before the loss
scaler and before the concept's own `loss_weight` — so `loss_weight` still scales
it exactly as it scales a positive concept.

The shape is [NPO](https://arxiv.org/pdf/2404.05868)'s, transplanted into
velocity space the same way [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
transplants DPO. Four properties follow, each pinned by a test in
`tests/test_counterexample_objective.py`:

- **It switches itself off.** `dL/dΔ = 2·sigmoid(β·Δ)`, so once the adapter is
  meaningfully worse than the reference on the bad image the gradient vanishes.
- **Its slope is bounded by 2** — one bad row cannot dominate a step.
- **Step 0 is exact.** A LoRA starts at zero ⇒ `v_θ ≡ v_ref` ⇒ `Δ = 0` ⇒
  `L = (2/β)·log 2` with exactly half-scale gradient. No spike.
- **Two knobs, and they do different jobs.** `β` sets the *scale*;
  `counterexample_ramp` sets the *timing*.

## The alternative it exists to replace

`concept.loss_weight` is an unclamped float, so `−1` is already accepted today
and is naive gradient ascent on an unbounded loss. In the toy end-to-end test in
this repo (same adapter, same steps, same learning rate) the bounded term settles
at **2.3× the reference distance**; ascent reaches **`inf`**. That is the whole
argument for the bound.

## β sets the *scale*, the ramp sets the *timing*

These are two different knobs and the difference is the thing to get right.

**β cannot make the term start gently.** `dL/dΔ = 2·sigmoid(β·Δ)`, so at `Δ = 0`
— exactly where every cold LoRA starts — the slope is `1.0` for *every* β. All β
moves is the `Δ` scale at which the bound engages. Ramping β up over training is
in fact an *anti*-ramp: small β early is the regime where the gate sits at 0.5 and
never switches off (constant-rate repulsion — the unbounded ascent this replaces),
and large β late is where it finally bounds.

### `counterexample_beta` — leave it at 0

`0` is the default and means **solve it from this run's own Δ**. The
hundreds-to-thousands range quoted from Diffusion-DPO's `β_T` convention did not
survive a real run: on SD 1.5 LoRA, `β = 1000` gave `β·|Δ| ≈ 0.015` and wanted
**~31,500**. `|Δ|` also *grows as the adapter trains* — 21× within one 48-step
run — so no published constant is right for every model, resolution and stage.

With `0`, the schedule holds a bootstrap β for the first **25% of the run** while
it measures `|Δ|`, then solves `sigmoid(β·|Δ|) = 0.9` once and **freezes** it.
Frozen rather than tracked: a β that kept chasing `|Δ|` would pin the gate forever
and destroy the switch-off the bound exists for. The value in force is logged as
`counterexample/beta`, and it changes exactly once.

The window is its own fraction of the run rather than "the ramp window" — at
`counterexample_ramp = 1.0` the ramp only closes on the last step, which would
calibrate β exactly as the run ends. At the `0.25` ramp default the two coincide.

⚠️ **If `counterexample/beta` never leaves 1000, β never calibrated.** A bootstrap
β is indistinguishable from an explicitly configured 1000, so check the tag rather
than assuming. It can only happen when a run has fewer than 8 steps carrying a
counterexample row at all.

### `counterexample_ramp` — the strength schedule

The fraction of the run (or, above 1, a literal step count) over which the term
eases 0 → full on a raised cosine. `0` disables it. This is the knob that makes
the counterexample term's timing independent of the positives': it stops the run
repelling from a wrong image *before the adapter has learned anything*, which is
the cold-start objection β cannot answer.

At the default it coincides with the β calibration window, so the gentle start
pays for itself twice.

| `counterexample_ramp` | mean strength over the run | full strength from |
|---|---|---|
| `0` | 1.00 | step 0 |
| `0.25` *(default)* | 0.87 | 25% in |
| `0.5` | 0.75 | halfway |
| `1.0` | 0.50 | the last step |

**Default 0.25, not 1.0, and the difference is a real trade.** A cosine ramp
across the whole run is the right shape for "strongest during the LR anneal", but
it delivers *half* the total repulsion — as a default that would halve the
treatment in an A/B and could report a real effect as a null. Set `1.0`
deliberately when concentrating the correction into the anneal is the goal.

### `counterexample_band_low` / `counterexample_band_high` — *where* on the schedule

The third knob, and the one that decides **which noise levels** the repulsion is
allowed to touch. `0`/`1` is no band and is exactly a no-op — the shipped default.

The band is expressed in

```
u = 1 / (1 + sqrt(SNR))
```

— the fraction of the noised latent's **amplitude** that is noise. `u = 0` is a
clean latent, `u = 1` is pure noise. For a variance-preserving schedule
(`x_t = sqrt(a_bar)·x_0 + sqrt(1-a_bar)·eps`) that is
`sqrt(1-a_bar) / (sqrt(a_bar) + sqrt(1-a_bar))`; for a rectified flow
(`x_t = (1-sigma)·x_0 + sigma·eps`) the same expression collapses to **`sigma`
exactly**.

**⚠️ It is deliberately not the coordinate `min_noising_strength` uses.**
Timestep-index fraction is not comparable across model families: on SD 1.5's
scaled-linear schedule `t/N = 0.1` is already `u = 0.26`, while on a
rectified-flow model `t/N = 0.1` is `u = 0.10`. A band authored on one model and
reused on another would silently cover a different physical noise range, and
nothing would report it. That is the whole reason this knob exists in its own
coordinate.

It is a **reweighting, not a resampling**. The timestep is drawn per sample
before concept type is ever consulted, so narrowing `min/max_noising_strength`
instead would move the positives' schedule too.

**Why a band at all.** Diffusion training decomposes into
[three stages by noise level](https://arxiv.org/pdf/2204.00227) — coarse
structure at low SNR, perceptually-rich *content* in the middle, imperceptible
clean-up at high SNR. A counterexample is a *close-but-wrong* image, so what
separates it from a right one is content, not global structure. The unlearning
literature reaches the same place from the other side:
[KSCU](https://arxiv.org/html/2507.06526) confines concept unlearning to the last
70% of denoising steps and reports FID **14.1** against **18.8** for training on
all steps, with the high-noise 30% producing "severe structural collapse" (FID
69.7) when intervened on. In `u`, that 70% is roughly `u <= 0.77` on SD 1.5.

Measured on a 384-step SD 1.5 LoRA run with the band off and every row's `(u, Δ)`
dumped, mean `|Δ|` by band is **1.00 / 0.65 / 0.31 / 0.26 / 0.24** from clean to
noisy — monotone, and the ordering holds within each third of training. Keeping
`u <= 0.77` retains **82% of the Δ mass for 71% of the rows**. That is one dataset
and one model, which is why it ships **defaulted off**; `0` / `0.77` is the
documented starting point, not a recommendation with evidence behind it yet.

β calibrates on the rows the band lets through, not on all of them: `|Δ|` varies
strongly with noise level, so solving on rows the band removed would target a
scale the run never optimizes. A step muted entirely contributes nothing rather
than a zero.

**⚠️ Narrowing the band reduces the DOSE.** A band that passes 40% of rows
delivers 40% of the repulsion. `counterexample/band_pass` reports the fraction,
and two arms of an A/B with different bands are not the same treatment at the
same `loss_weight` — match the band, or divide `loss_weight` by the pass
fraction. The same trap as the ramp, in a different dimension.

The edges are raised cosines a quarter of the band's width each, combined with a
`min`, so every band keeps a full-strength plateau across its middle half rather
than degenerating into a spike.

## Telemetry

Logged once per optimizer step (aggregated over the gradient-accumulation
window), only on steps that actually contained counterexample rows:

| tag | read it as |
|---|---|
| `counterexample/gate_mean` | **read this first.** Mean `sigmoid(β·Δ)` — the fraction of full strength the term is running at. A cold LoRA starts at exactly `0.5`. `~1.0` is **not** a β fault: a gate near 1 requires `β·Δ ≥ 2.197` and so *proves* the bound engaged — it says `Δ` stayed positive and the term has not yet won. (β too small is a gate pinned at `0.5`.) `~0.0` is **ambiguous** — see below. |
| `counterexample/saturated_fraction` | fraction of rows with `Δ < 0` — strictly worse than the reference on that image. Rising from 0 = the job is being done. |
| `counterexample/weight` | the ramp's current strength, 0…1. A run that looks quiet early may simply be ramping. |
| `counterexample/beta` | the β actually in force — the solved one when `counterexample_beta = 0`. It changes exactly once, when the calibration window closes. |
| `counterexample/delta_mean` | mean `Δ`. Multiply by β to sanity-check the knob. |
| `counterexample/loss_mean` | mean `L`. Starts at `(2/β)·log 2`. |
| `counterexample/rows` | counterexample rows seen in the window. Zero means the concept never landed in a batch. |
| `counterexample/noise_level` | mean `u` of the counterexample rows the sampler drew, **before** the band filters them — i.e. where the rows are, which is what tells you where to put a band. Unbanded it is the same number either way. |
| `counterexample/band_pass` | the term's **dose**: mean band weight over its rows. `1.0` is unbanded. Far below it and the band has narrowed the term nearly out of existence — a failure `gate_mean` cannot see, because the rows that *do* pass behave perfectly normally. |

### ⚠️ `gate_mean ≈ 0` has two opposite meanings

It is the single most misread number in this feature. Near-zero says the term is
not currently pushing, and that happens for two reasons that no single reading
can tell apart:

1. **It worked.** The adapter has been driven past the reference on every
   counterexample row, so the objective has switched itself off. This is the
   designed end state.
2. **It never engaged.** The term was inert from the start and the run learned
   nothing from its counterexamples.

**Only the trajectory separates them.** A gate that started near `0.5` — where
every cold LoRA begins — and fell has done its work. A gate that was already near
zero on the first logged step never started, and `saturated_fraction` will have
been pinned high from step 0 rather than climbing.

The most common cause of case 2 is a **warm start**: see the note below.

## Requirements and costs

- **LoRA only.** The frozen reference is `prior_model()`, which detaches every
  adapter hook; it raises `NotImplementedError` for other training methods.
- **One extra full forward** whenever a batch contains a counterexample row — the
  same cost `PRIOR_PREDICTION` already pays, and shared with it when both are
  present. Batches without either are unaffected.
- **⚠️ On a warm start the reference is the *foundation*, not the adapter you
  resumed from.** `prior_model()` is all-or-nothing over `model.adapters()`. A
  strongly-trained reseed can be better than the foundation on every image,
  positive and negative alike, which drives `Δ < 0` everywhere and makes the term
  silent from the first step. This is the main reason `gate_mean` exists — check
  it before concluding anything about a warm-started run.
- A counterexample row with **no** reference prediction raises rather than
  falling through as a positive. That is deliberate: such a row would train the
  model *toward* the near-miss it was supposed to be pushed away from, which is
  the one failure of this feature a green run would otherwise never reveal.

## Inertness

`COUNTEREXAMPLE` is a new `ConceptType`, and no existing concept has it. Every
existing configuration takes the same code path it did before, produces the same
losses, and logs none of the tags above.
