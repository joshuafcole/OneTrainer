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
- **One knob**: `β`.

## The alternative it exists to replace

`concept.loss_weight` is an unclamped float, so `−1` is already accepted today
and is naive gradient ascent on an unbounded loss. In the toy end-to-end test in
this repo (same adapter, same steps, same learning rate) the bounded term settles
at **2.3× the reference distance**; ascent reaches **`inf`**. That is the whole
argument for the bound.

## Choosing β

`Δ` is a difference of **element-mean** distances, so it is small — order
1e-3…1e-1 on latents. The switch-off only means anything when `β·|Δ|` is order 1,
which puts β in the hundreds-to-thousands. Default: **1000**.

⚠️ **That default is a starting convention, not a tuned value.** It is borrowed
from Diffusion-DPO's `β_T` for the same element-mean convention. A single
measurement on an SD 1.5 LoRA wanted roughly 30× more, so treat 1000 as a place
to begin and not as an answer.

Do not guess it twice: run a short run and read `counterexample/gate_mean` off
tensorboard.

## Telemetry

Logged once per optimizer step (aggregated over the gradient-accumulation
window), only on steps that actually contained counterexample rows:

| tag | read it as |
|---|---|
| `counterexample/gate_mean` | **read this first.** Mean `sigmoid(β·Δ)` — the fraction of full strength the term is running at. A cold LoRA starts at exactly `0.5`. `~1.0` = β too small to ever switch off. `~0.0` is **ambiguous** — see below. |
| `counterexample/saturated_fraction` | fraction of rows with `Δ < 0` — strictly worse than the reference on that image. Rising from 0 = the job is being done. |
| `counterexample/delta_mean` | mean `Δ`. Multiply by β to sanity-check the knob. |
| `counterexample/loss_mean` | mean `L`. Starts at `(2/β)·log 2`. |
| `counterexample/rows` | counterexample rows seen in the window. Zero means the concept never landed in a batch. |

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
