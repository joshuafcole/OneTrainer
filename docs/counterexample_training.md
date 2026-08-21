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

## Choosing β

`Δ` is a difference of **element-mean** distances, so it is small — order
1e-3…1e-1 on latents. The switch-off only means anything when `β·|Δ|` is order 1,
which puts β in the hundreds-to-thousands, matching Diffusion-DPO's `β_T` for the
same convention. Default: **1000**.

Do not guess it twice. Run a short run and read `counterexample/gate_mean` off
tensorboard (below).

## Telemetry

Logged once per optimizer step (aggregated over the gradient-accumulation
window), only on steps that actually contained counterexample rows:

| tag | read it as |
|---|---|
| `counterexample/gate_mean` | **read this first.** Mean `sigmoid(β·Δ)` — the fraction of full strength the term is running at. `~1.0` = β too small to ever switch off. `~0.0` = **the term is inert and the run learned nothing from its counterexamples.** |
| `counterexample/saturated_fraction` | fraction of rows with `Δ ≤ 0` — already worse than the reference. Rising toward 1 = the job is being done; pinned at 1 from step 0 = the reference is the wrong anchor. |
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
