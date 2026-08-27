# Concept Sliders

A **slider** is an ordinary LoRA (or LoHa/LoKr) trained so that its *multiplier*
becomes a continuous, signed control knob: at `0.0` the model is untouched, at
`+1.0` a concept is dialled up, at `-1.0` it is dialled down. It is the method of
Gandikota et al., *Concept Sliders: LoRA Adaptors for Precise Control in Diffusion
Models* (ECCV 2024).

There are two ways to train one, chosen with **Regime** on the Slider tab:

- **Prompt pair** — no image dataset at all. The frozen base model supplies the
  direction to move in, so all you write is prompts. This is the paper's method
  and what most of this page describes.
- **Image (coordinate)** — an ordinary image dataset whose captions say where each
  image sits on the axis. See [Coordinate-labeled image
  sliders](#coordinate-labeled-image-sliders) below. Anima only, for now.

Pick `Slider` in the training-method dropdown. It is offered for the model types
that have a slider host — currently Anima and SDXL.

## What it trains

For one prompt triple — a neutral `target`, and the same idea with more
(`positive`) and less (`negative`) of an attribute — the training target is

```
v*(x_t, c_target, t)  =  v(c_target)  +  eta * mean_p( v(c_positive, p) - v(c_negative, p) )
```

Everything on the right is computed by the **frozen base with the adapter
switched off**, so it is a fixed target the adapter is fitted to; only the
adapter's own weights move. With `Symmetric` on, the `-strength` pole is trained
toward the same expression with a minus sign, which is what keeps the slider
roughly linear around 0 instead of doing something unrelated at negative values.

`v` is written as a velocity above. Epsilon-parameterised models (SD/SDXL) need no
separate objective: at a fixed `(x_t, t)` the two differ by an affine map in `t`,
so the guidance *difference* is the same direction either way.

## The Slider tab

| control | what it does |
|---|---|
| **Regime** | `Prompt pair` is the text-only method described here; `Image (coordinate)` is [the other one](#coordinate-labeled-image-sliders). |
| **prompt pair list** | The triples. `target` is what you will be generating; `positive`/`negative` are the two poles. `weight` mixes several triples into one slider; `0` excludes one without deleting it. |
| **Guidance scale** | `eta` above. How far past the base model the target is pushed. |
| **Training strength** | The adapter multiplier the trained passes run at (`+/-` this). |
| **Symmetric** | Train the negative pole too. One extra forward per step. |
| **Steps per epoch** | There is no dataset, so this is what an "epoch" means here — it is what backups, sampling and the LR schedule count against. |
| **Latent anchor steps** | Denoising steps used to walk down to the sampled noise level under the target prompt, so the training latent sits on the model's own trajectory. `0` uses plain Gaussian noise: much cheaper, but it measures the guidance direction somewhere the model never actually visits. |
| **Sigma min / max** | The noise range the trained timestep is drawn from. Either order. |
| **Preservation prompts** | Optional. Contexts in which the attribute should change and *nothing else* should, separated by `\|` or newlines. Each is appended to both poles and the guidance direction is averaged over all of them (the paper's disentanglement term). Leave empty for the bare pair. |

`eta` and the inference-time slider strength are **independent**. `eta` only sets
how strong a direction is baked into the adapter during training; how far a user
pushes the slider afterwards is the multiplier they load it at. If in doubt leave
`eta` low: too small a value gives a weak slider that can still be pushed past
`1.0` at inference, while too large a value bakes artifacts in that no inference
multiplier can take back out.

The adapter is left at multiplier `1.0` between steps, so samples generated during
training show the slider at the strength most inference tools apply by default.

## The rest of the UI

The **LoRA tab** configures the network itself — rank, alpha, PEFT type, layer
filter — exactly as for a normal LoRA run, and the file it saves is a normal LoRA.

The adapter must have a signed multiplier to slide, so **DoRA, OFT and
weight-decomposed LoKr cannot be used**: they recompose the base weight rather
than adding a scaled delta. Choosing one is refused when the run starts, not
halfway through.

In the prompt-pair regime the **Concepts tab is unused.** So is everything downstream of it: prior
preservation, masked training, counterexamples and the loss-weighting controls all
select rows out of a dataset, and a prompt-pair slider has none. Validation
likewise reports nothing rather than resampling the training distribution and
calling the result a metric.

## Coordinate-labeled image sliders

The prompt-pair regime needs the base model to already know the attribute — it
can only push toward something it can be asked for in words. When it does not,
you can show it instead.

In the `Image (coordinate)` regime you train on an **ordinary dataset from the
Concepts tab**. No paired files, no new format: what makes it a slider is that
each caption says where that image sits on the axis, written the way a1111 writes
emphasis.

```
a photo of a car on a road, (distance:-2)
a photo of a car on a road, (distance:2)
```

Declare `distance` under **Coordinate axes** on the Slider tab. That token is then
taken out of the caption before training and used as the adapter multiplier
instead: the image labelled `-2` has to be reconstructed with the adapter at
`gain * -2`, the one labelled `2` at `gain * 2`. The only way to satisfy every
image at once is for the adapter's effect to scale with its multiplier — which is
exactly the knob you turn afterwards.

Emphasis tokens you did **not** declare, like `(red car:1.2)`, are left alone, so
an already-captioned dataset keeps working.

### Getting the captions right

Removing the coordinate from the caption is the load-bearing part, and it decides
what the rest of the caption has to say. The caption should describe everything
about the image **except** the axis. That is what leaves the axis as the only
thing the adapter has left to explain. If the captions still said "close up" and
"far away", the base could read the attribute off the prompt and the adapter would
learn nothing; if the captions were blank, the adapter would start absorbing image
content instead of the axis.

Caption dropout is therefore **off** in this regime, whatever the Training tab
says, for the same reason.

Some practical consequences:

- **Spread the labels either side of 0.** That is what gives you a slider that
  runs both ways. Labels of only `-1` and `+1` are fine and give you two poles;
  more values in between give you a calibrated axis.
- **An image labelled `0` trains nothing** and is dropped from the step: at
  multiplier 0 the adapter is switched off, so there is no gradient to be had. A
  batch in which *every* coordinate is 0 stops the run, because that is what a
  mistyped axis name looks like.
- **Label in whatever units you measured.** Values are read as written, and
  `gain` is what brings the extremes to roughly `1.0`. With labels spanning
  `-2..2`, a gain of `0.5` trains the extremes at the multiplier inference applies
  by default.
- **Confounders can be declared without being trained.** An axis that is not the
  target is still stripped from the captions, which keeps something you have
  labelled out of the conditioning on a run that is not about it. Exactly one
  enabled axis is the target.

### What carries over and what does not

`Sigma min / max` still set the noise range of the trained timestep. Everything
else in the prompt-pair block does not apply and is hidden: there is no guidance
scale, no training strength (the coordinate is the strength), no symmetric pole,
no steps-per-epoch (the dataset has a length), and no latent anchor walk (the
image is already on the manifold).

The Concepts tab, aspect bucketing, latent and text caching all work normally.
`gain` is read at training time, so retuning it does not invalidate the cache.

> **Maturity.** This regime is newer than the prompt-pair one and has had far less
> mileage. The caption handling, the multiplier schedule and the step are covered
> by tests; a long real training run is the part that is thin.

## Using the result

Load the saved file as an ordinary LoRA and set its strength. Negative strengths
are meaningful here, which is the whole point — `-1.0` is the other end of the
slider, not "off".
