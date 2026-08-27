# Concept Sliders

A **slider** is an ordinary LoRA (or LoHa/LoKr) trained so that its *multiplier*
becomes a continuous, signed control knob: at `0.0` the model is untouched, at
`+1.0` a concept is dialled up, at `-1.0` it is dialled down. It is the method of
Gandikota et al., *Concept Sliders: LoRA Adaptors for Precise Control in Diffusion
Models* (ECCV 2024).

The distinctive part is that **there is no image dataset**. The frozen base model
supplies the direction to move in, so all you write is prompts.

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
| **Regime** | `Prompt pair` is the text-only method described here. |
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

The **Concepts tab is unused.** So is everything downstream of it: prior
preservation, masked training, counterexamples and the loss-weighting controls all
select rows out of a dataset, and a prompt-pair slider has none. Validation
likewise reports nothing rather than resampling the training distribution and
calling the result a metric.

## Using the result

Load the saved file as an ordinary LoRA and set its strength. Negative strengths
are meaningful here, which is the whole point — `-1.0` is the other end of the
slider, not "off".
