# Counterexample Training

Train a concept **away from** close-but-wrong images, alongside the positives it
is trained toward.

Set a concept's **Type** to `COUNTEREXAMPLE` in the Concepts tab. Its images are
near-misses — renders from a previous run that a human marked bad — captioned
**with the same trigger as the positives**. A near-miss captioned *without* the
trigger is a different experiment (concept ablation via an anchor), not this one.

## What it does

Rectified flow gives target velocity `v = ε − x₀`. For each counterexample row,
at the same `(x_t, t)` the positive rows use:

```
d      = ‖ v_θ(x_t, c, t) − v ‖²     trained  (adapter on)
d_ref  = ‖ v_ref(x_t, c, t) − v ‖²   frozen   (adapter off)
Δ      = d_ref − d      # > 0  ⇔  the adapter fits the WRONG image
                        #          BETTER than the base model does
L      = (2/β) · softplus(β · Δ)
```

The row's ordinary reconstruction loss is **replaced** by `L`, before the loss
scaler and before the concept's own `loss_weight` — so `loss_weight` still ramps
it exactly as it ramps a positive concept.

The shape is [NPO](https://arxiv.org/pdf/2404.05868)'s, transplanted into
velocity space the same way [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
transplants DPO. Four properties follow, each pinned by a test in
`tests/test_counterexample_objective.py`:

- **It switches itself off.** `dL/dΔ = 2·sigmoid(β·Δ)`, so once the adapter is
  meaningfully worse than the reference on the bad image the gradient vanishes.
- **Its slope is bounded by 2** — one bad row cannot dominate a step.
- **Step 0 is exact.** A LoRA starts at zero ⇒ `v_θ ≡ v_ref` ⇒ `Δ = 0` ⇒
  `L = (2/β)·log 2` with exactly half-scale gradient. No spike.
- **One knob**: `β`.

## The alternative it exists to replace

`concept.loss_weight` is an unclamped float, so `−1` is already accepted and is
naive gradient ascent on an unbounded loss. In the toy end-to-end test in this
repo (same adapter, same steps, same learning rate) the bounded term settles at
**2.3× the reference distance**; ascent reaches **`inf`**. That is the whole
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

`0` means **solve it from this run's own Δ**, and that is the recommendation.
The hundreds-to-thousands range quoted from Diffusion-DPO's `β_T` convention did
not survive a real run: on SD 1.5 LoRA, `β = 1000` gave `β·|Δ| ≈ 0.015` and wanted
**~31,500**. `|Δ|` also *grows as the adapter trains* — 21× within one 48-step run
— so no published constant is right for every model, resolution and stage.

With `0`, the schedule holds a bootstrap β for the first **25% of the run** while
it measures `|Δ|`, then solves `sigmoid(β·|Δ|) = 0.9` once and **freezes** it.
Frozen rather than tracked: a β that kept chasing `|Δ|` would pin the gate forever
and destroy the switch-off the bound exists for. The value used is logged as
`counterexample/beta`, and it changes exactly once.

The window is its own fraction of the run rather than "the ramp window" — at
`counterexample_ramp = 1.0` the ramp only closes on the last step, which would
calibrate β exactly as the run ends. At the `0.25` ramp default the two coincide.

⚠️ **If `counterexample/beta` never leaves 1000, β never calibrated.** A bootstrap
β is indistinguishable from an explicitly configured 1000, so check the tag rather
than assuming. It can only happen now when a run has fewer than 8 steps carrying a
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

## Telemetry

Logged once per optimizer step (aggregated over the gradient-accumulation
window), only on steps that actually contained counterexample rows:

| tag | read it as |
|---|---|
| `counterexample/gate_mean` | **read this first.** Mean `sigmoid(β·Δ)` — the fraction of full strength the term is running at. `~1.0` = `Δ` stayed positive, so the term never won (NOT a β fault: a gate near 1 proves `β·Δ` is large). `~0.0` = **the term is inert and the run learned nothing from its counterexamples.** |
| `counterexample/saturated_fraction` | fraction of rows **strictly** past the reference (`Δ < 0`). Strictly, because a cold LoRA's step 0 is `Δ = 0` for every row and counting it would report "all done" before anything happened. |
| `counterexample/weight` | the ramp's current strength, 0…1. A run that looks quiet early may simply be ramping. |
| `counterexample/beta` | the β actually in force — the solved one when `counterexample_beta = 0`. It changes exactly once, when the calibration window closes. |
| `counterexample/delta_mean` | mean `Δ`. Multiply by β to sanity-check the knob. |
| `counterexample/loss_mean` | mean `L`. Starts at `(2/β)·log 2`. |
| `counterexample/rows` | counterexample rows seen in the window. Zero means the concept never landed in a batch. |

## Requirements and costs

- **LoRA only.** The frozen reference is `prior_model()`, which detaches every
  adapter hook; it raises `NotImplementedError` for other training methods.
- **~1.6–1.9× step time** whenever a batch contains a counterexample — one extra
  full forward, the same cost `PRIOR_PREDICTION` already pays. Batches without
  one are unaffected.
- **⚠️ On a warm start the reference is the *foundation*, not the adapter you
  resumed from.** `prior_model()` is all-or-nothing on `model.adapters()`. A
  strongly-trained reseed can be better than the foundation on every image,
  positive and negative alike, which drives `Δ ≤ 0` everywhere and makes the term
  silent. This is not hypothetical — it is why `gate_mean` exists. Check it
  before concluding anything about a warm-started run.
- Counterexample rows are excluded from the kron-GA (`LokrInitMode`) estimation
  pass, like `PRIOR_PREDICTION` rows, since their step-0 gradient points away
  from the data and would pollute the gradient-alignment estimate.
- A counterexample row with **no** reference prediction raises rather than
  falling through as a positive — the one failure of this feature a green run
  would otherwise never reveal.
