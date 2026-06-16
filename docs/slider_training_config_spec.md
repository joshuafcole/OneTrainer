# Slider LoRA Training — Configuration & Handoff Spec

**Audience:** an agent building the cinema-studio web UI for configuring and managing
LoRA *slider* training runs against this OneTrainer (OT) fork. Self-contained — no OT
internals are assumed. It describes the two training modes, every setting that needs
to be configured for each, the dataset contract, validation rules, and the
input/output artifacts so the web UI can render forms, emit a valid run config, launch
it, and surface the result.

> Status: the training code is implemented and unit-tested; first end-to-end GPU
> validation is still pending, so the numeric *recommendations* below are starting
> points (transferred from the Concept-Sliders literature), not yet tuned on this
> model. The *schema* (field names, types, which-applies-to-which) is authoritative.

---

## 1. What a slider LoRA is

A **slider** is an ordinary LoRA adapter whose **signed multiplier is a continuous
control knob** at inference time:

```
output(m) = base + m · Δ(LoRA)      m ∈ [−strength … 0 … +strength]
```

- `m = 0` → the base model's behavior (the slider is off).
- `m > 0` / `m < 0` → push the target attribute toward one pole / the other.

Training teaches the adapter the *direction* of one attribute axis. At inference the
user scrubs `m`. This is different from a normal LoRA (which has a single baked-in
effect); a slider is explicitly trained to be *linear and reversible* around 0.

**Model support:** only **Anima** is wired today. The web UI must set
`model_type = "ANIMA"`; no other model accepts the `SLIDER` training method yet.

---

## 2. The two training modes

| | **Prompt-pair** (`PROMPT_PAIR`) | **Image (coordinate)** (`IMAGE`) |
|---|---|---|
| Image dataset? | **No** (text-only, synthetic) | **Yes** (real images + captions) |
| How the attribute is defined | positive/negative **prompt pairs** | a per-image **caption coordinate** |
| Mechanism | frozen base supplies the guidance direction `v(c+) − v(c−)`; adapter learns it | each image reconstructed at multiplier `m = k·coordinate` (coordinate-scaled reconstruction) |
| Use when | you can *describe* the attribute in words (age, lighting, style) | you have *labeled imagery* spanning the axis (distance, count, a measured quantity) |
| Confound control | preservation prompts (optional) | balanced/symmetric dataset (manual in v1) |

A run is **one slider = one attribute axis**. To ship multiple sliders, configure and
launch multiple runs.

---

## 3. How configs are delivered to OT

OT is driven by a **train-config JSON** (flat keys; each key is a config field; enums
are their string value; list fields are arrays of objects). The dataset is supplied
either inline (`concepts`) or — recommended for programmatic use — referenced by path
(`concept_file_name` → a `concepts.json`). Sample prompts likewise
(`sample_definition_file_name` → `samples.json`).

> **Strong recommendation for the web UI:** do **not** hand-build the full config from
> scratch — OT's `TrainConfig` has hundreds of fields with cross-field defaults and
> validation. Instead start from a **known-good template** (a config exported from the
> OT desktop UI for an Anima LoRA, or OT's serialized defaults) and **override only the
> keys in this doc.** Leave everything else at the template's values. Round-tripping a
> template guarantees the nested sub-configs (optimizer, cloud, embeddings, per-concept
> image/text blocks) stay schema-valid.

The keys in §4–§8 are the ones the web UI must surface and set for slider runs.

---

## 4. Base settings — required for **both** modes

These are standard OT fields (not slider-specific) but must be set for a slider run.

| JSON key | Type | Slider value / recommendation | Notes |
|---|---|---|---|
| `training_method` | enum str | **`"SLIDER"`** | selects the slider trainer |
| `model_type` | enum str | **`"ANIMA"`** | only wired model |
| `base_model_name` | str | path/HF id of the Anima base | the model to adapt |
| `output_model_destination` | str | e.g. `"models/my_slider.safetensors"` | where the LoRA is written |
| `output_model_format` | enum str | `"SAFETENSORS"` | |
| `peft_type` | enum str | `"LORA"` (recommended) | also `"LOKR"`, `"LOHA"`, `"OFT_2"` |
| `lora_rank` | int | **4–8** (sliders want low rank) | OT default 16 is high for sliders |
| `lora_alpha` | float | ≈ `rank/2` | effective scale ≈ `alpha/rank` |
| `layer_filter_preset` | str | `"full"` \| `"blocks"` \| `"detail"` \| `"attn-mlp"` \| `"attn-only"` \| `"cross-attn"` | which modules the adapter targets; `attn-only`/`cross-attn` localize the effect |
| `layer_filter` | str | `""` | custom layer filter (overrides preset when set) |
| `learning_rate` | float | ~`1e-4`–`1e-3` (starting point) | sliders converge fast |
| `optimizer` | object | AdamW or Prodigy | leave sub-config from template; set `.optimizer` name |
| `epochs` | int | small (sliders need few steps) | combine with `slider_steps_per_epoch` for PROMPT_PAIR |
| `batch_size` | int | 1–4 | per-sample multiplier means one forward per sample |
| `resolution` | str | e.g. `"512"` or `"512x768"` | training resolution; comma-list = multi-res buckets |
| `train_dtype` | enum str | `"FLOAT_16"` / `"BFLOAT_16"` | |
| `gradient_checkpointing` | enum str | `"ON"` | |

**Optional — gradient-aligned (GA) init** (pre-orients the adapter along the attribute
direction; speeds convergence; works for LoRA and LoKr):

| JSON key | Type | Value | Notes |
|---|---|---|---|
| `lokr_init_mode` | enum str | `"GRADIENT"` to enable (else `"DEFAULT"`) | despite the `lokr_` name it applies to LoRA too |
| `lokr_init_steps` | int | e.g. 64 | gradient-estimation steps for the init |
| `lokr_init_gain` | float | 1.0 | scale of the init |
| `lokr_init_offload` | bool | false | offload during init to save VRAM |

---

## 5. Prompt-pair mode settings (`slider_regime = "PROMPT_PAIR"`)

No dataset. The attribute is defined entirely by prompt triples.

| JSON key | Type | Default | Meaning |
|---|---|---|---|
| `slider_regime` | enum str | `"PROMPT_PAIR"` | selects this mode |
| `slider_prompts` | array | `[]` | the attribute triples (≥1 enabled required) |
| `slider_preservation_prompts` | str | `""` | optional disentanglement set, **pipe-delimited** `a | b | c`; empty = bare pair |
| `slider_eta` | float | `3.0` | training-time guidance scale on `v(c+) − v(c−)` (paper ~3–4) |
| `slider_strength` | float | `1.0` | adapter multiplier magnitude used during training (decoupled from the inference range) |
| `slider_symmetric` | bool | `true` | also train the `−strength` pole → slider is linear around 0 |
| `slider_steps_per_epoch` | int | `500` | synthetic step count per epoch (no dataset drives it) |
| `slider_anchor_steps` | int | `8` | Euler steps to put the noised latent on-manifold (SDEdit); 0 = cheaper off-manifold |
| `slider_sigma_min` | float | `0.1` | low bound of the sampled noise level |
| `slider_sigma_max` | float | `0.9` | high bound (higher σ concentrates the attribute signal) |

**`slider_prompts[]` element** (`SliderPromptConfig`):

| field | type | default | meaning |
|---|---|---|---|
| `enabled` | bool | `true` | include this triple |
| `target` | str | `"a portrait photo of a person"` | the **neutral** concept `c_t` the slider centers on |
| `positive` | str | `"a portrait photo of an old person"` | **+** pole conditioning `c+` |
| `negative` | str | `"a portrait photo of a young person"` | **−** pole conditioning `c−` |
| `weight` | float | `1.0` | sampling weight when mixing several triples |
| `uuid` | str | (generate) | stable id; generate a fresh UUID per element |

**Validation:** at least one `enabled` triple. `target/positive/negative` non-empty.

---

## 6. Image (coordinate) mode settings (`slider_regime = "IMAGE"`)

Uses a **real image dataset** (see §7). The attribute is defined by per-image caption
coordinates. **Unused in this mode** (leave at default): `slider_prompts`,
`slider_preservation_prompts`, `slider_eta`, `slider_strength`, `slider_symmetric`,
`slider_anchor_steps`, `slider_steps_per_epoch` (epoch length is driven by the dataset).

| JSON key | Type | Default | Meaning |
|---|---|---|---|
| `slider_regime` | enum str | `"IMAGE"` | selects this mode |
| `slider_axes` | array | `[]` | declared coordinate axes (see element below) |
| `slider_sigma_min` | float | `0.1` | low bound of the sampled noise level |
| `slider_sigma_max` | float | `0.9` | high bound |

**`slider_axes[]` element** (`SliderAxisConfig`):

| field | type | default | meaning |
|---|---|---|---|
| `enabled` | bool | `true` | include this axis |
| `name` | str | `"distance"` | the caption token key, e.g. `distance` for `(distance:-2)`; matched case-insensitively and **stripped from the caption** |
| `gain_k` | float | `1.0` | multiplier gain: `m = gain_k · coordinate` |
| `is_target` | bool | `true` | **exactly one** enabled axis must be the target — its coordinate drives the slider |
| `stratify` | bool | `false` | reserved for the balanced sampler (roadmap; ignored in v1) |
| `uuid` | str | (generate) | stable id |

**How the coordinate maps to the slider:** a caption like
`a car on a road, (distance:-2)` yields coordinate `−2`; with `gain_k = 1` the adapter
is trained at multiplier `m = −2` to reconstruct that image. Binary "before/after"
data is just the coordinates `{−1, +1}`. After training, the inference knob is
calibrated in the same units (× `gain_k`).

**Validation:**
- exactly **one** `enabled` axis with `is_target = true`;
- at least one enabled concept dataset (§7);
- recommend a **coordinate-symmetric** dataset (roughly balanced + / − / 0) so the
  slider stays centered — in v1 this is the data author's responsibility.

---

## 7. Dataset / caption contract (IMAGE mode only)

The dataset is **vanilla OT concepts** — the same structure used for any LoRA. The
*only* slider-specific requirement is the caption annotation. Supply via
`concept_file_name` → a `concepts.json` array, or inline `concepts`.

**Per-concept fields that matter for sliders** (leave the rest at template defaults):

| key (within a concept object) | value | notes |
|---|---|---|
| `name` | str | label |
| `path` | str | folder of images |
| `enabled` | bool `true` | |
| `type` | `"STANDARD"` | not a regularization/prior concept |
| `include_subdirectories` | bool | recurse the folder |
| `text.prompt_source` | `"sample"` \| `"filename"` \| `"concept"` | where captions come from: `sample` = per-image `.txt` sidecar (recommended), `concept` = one shared caption file, `filename` = the filename |
| `text.prompt_path` | str | caption file path when `prompt_source = "concept"` |
| `text.tag_dropout_enable` | **`false`** | **must be off** — dropout could eat the coordinate token |

**Caption authoring rules (v1 — manual dataset prep):**
1. Embed the coordinate as an a1111-style token: `… , (axisname:value)` where
   `axisname` matches a declared `slider_axes[].name`. Example: `(distance:-2)`.
   Ordinary emphasis like `(red car:1.2)` is left untouched.
2. **Keep attribute prose out of the caption.** Do not also write "far away" /
   "close up" — only the coordinate token. If the caption names the attribute the
   slider entangles and weakens. (Auto-stripping a phrase list is a roadmap item.)
3. **Pre-scale the coordinates** yourself (ordinal recommended, e.g. −2…+2, symmetric
   around 0; continuous/measured values also fine, tuned via `gain_k`). Per-axis
   auto-rescale is a roadmap item.
4. Every training image should carry the target axis token. A missing coordinate is
   treated as `0` (that image becomes a no-op for the adapter — harmless but wasted).

---

## 8. Sampling & monitoring (optional, both modes)

| key | type | notes |
|---|---|---|
| `sample_definition_file_name` | str | → `samples.json` (array of sample-prompt configs) for previews during training |
| `samples` | array/null | inline alternative to the file |
| `sample_after` / `sample_after_unit` | int / enum str (`"MINUTE"`,`"EPOCH"`,`"STEP"`…) | preview cadence |
| `samples_to_tensorboard` | bool | log previews to TensorBoard |

> **Slider preview caveat:** during-training samples currently render with the
> adapter at its **last-set multiplier**, not a swept range — so they are *not* a true
> slider preview. For a real preview, generate at several multipliers (`−s, 0, +s`)
> **after** the run using the output LoRA. (Multiplier-sweep sampling is on the
> roadmap.) The web UI's "preview a slider" feature should drive post-train inference
> at multiple multipliers rather than rely on in-training samples.

Backups (`backup_*`) and TensorBoard (`tensorboard*`) are standard OT fields — leave
them on the template defaults or expose as generic run options.

---

## 9. Output artifact & inference

A run produces a **standard LoRA safetensors** at `output_model_destination`. There is
nothing slider-specific about the file — the "slider" is how you *use* it:

- Load the LoRA into any Anima inference path and set its **weight / multiplier** to
  the desired `m`. Negative ↔ positive sweeps the attribute; `0` (or LoRA disabled) =
  base.
- Training `slider_strength` / `slider_eta` are **decoupled** from the usable
  inference range — you can push `m` beyond the training strength (with rising risk of
  artifacts). For coordinate sliders, `m` is calibrated in `coordinate × gain_k` units.

The web UI's "manage runs" surface should therefore track, per run: the config used,
the regime, the target axis / prompt triples, the output safetensors path, and a
recommended starting multiplier range (e.g. ±`slider_strength` for prompt-pair, or the
coordinate range × `gain_k` for image).

---

## 10. Validation rules the web UI should enforce (pre-launch)

- `training_method == "SLIDER"` ⇒ `model_type == "ANIMA"`.
- `peft_type` ∈ {LORA, LOKR, LOHA, OFT_2}; recommend LORA, low `lora_rank` (4–8).
- **PROMPT_PAIR:** ≥ 1 `enabled` `slider_prompts` entry, each with non-empty
  `target`/`positive`/`negative`. (`slider_axes` and the dataset are ignored.)
- **IMAGE:** exactly one `enabled` `slider_axes` entry with `is_target == true`;
  ≥ 1 `enabled` concept with a valid `path`; per-concept `text.tag_dropout_enable ==
  false`. (`slider_prompts` etc. are ignored.)
- `slider_sigma_min < slider_sigma_max`, both in `(0, 1)`.
- Generate fresh `uuid`s for new `slider_prompts` / `slider_axes` / concept elements.

> OT will also raise a clear runtime error at launch if a slider run is misconfigured
> (no enabled triple / not exactly one target axis), but the web UI should catch these
> first.

---

## 11. Minimal config examples

**Prompt-pair** (overrides on top of a known-good Anima-LoRA template):

```json
{
  "training_method": "SLIDER",
  "model_type": "ANIMA",
  "base_model_name": "<anima base path or id>",
  "output_model_destination": "models/age_slider.safetensors",
  "output_model_format": "SAFETENSORS",
  "peft_type": "LORA",
  "lora_rank": 8,
  "lora_alpha": 4.0,
  "layer_filter_preset": "attn-only",
  "learning_rate": 0.0005,
  "epochs": 1,
  "batch_size": 1,
  "resolution": "512",

  "slider_regime": "PROMPT_PAIR",
  "slider_prompts": [
    {
      "uuid": "<uuid>", "enabled": true, "weight": 1.0,
      "target": "a portrait photo of a person",
      "positive": "a portrait photo of an old person",
      "negative": "a portrait photo of a young person"
    }
  ],
  "slider_preservation_prompts": "a man | a woman",
  "slider_eta": 3.0,
  "slider_strength": 1.0,
  "slider_symmetric": true,
  "slider_steps_per_epoch": 500,
  "slider_anchor_steps": 8,
  "slider_sigma_min": 0.1,
  "slider_sigma_max": 0.9
}
```

**Image (coordinate)** (overrides + a concepts file):

```json
{
  "training_method": "SLIDER",
  "model_type": "ANIMA",
  "base_model_name": "<anima base path or id>",
  "output_model_destination": "models/distance_slider.safetensors",
  "output_model_format": "SAFETENSORS",
  "peft_type": "LORA",
  "lora_rank": 8,
  "lora_alpha": 4.0,
  "layer_filter_preset": "full",
  "learning_rate": 0.0003,
  "epochs": 20,
  "batch_size": 2,
  "resolution": "512",
  "concept_file_name": "training_concepts/distance.json",

  "slider_regime": "IMAGE",
  "slider_axes": [
    { "uuid": "<uuid>", "enabled": true, "name": "distance", "gain_k": 1.0, "is_target": true, "stratify": false }
  ],
  "slider_sigma_min": 0.1,
  "slider_sigma_max": 0.9
}
```

**`training_concepts/distance.json`** (each image has a `.txt` sidecar caption such as
`a car on a road, (distance:-2)`):

```json
[
  {
    "name": "distance dataset",
    "path": "datasets/distance",
    "enabled": true,
    "type": "STANDARD",
    "include_subdirectories": true,
    "text": { "prompt_source": "sample", "prompt_path": "", "tag_dropout_enable": false }
  }
]
```

> Emit the **full** concept object via the template (it has `image`/`text`/etc.
> sub-blocks); the keys above are the slider-relevant ones — keep the rest at OT
> defaults.

---

## 12. Roadmap (not yet available — don't build UI against these)

- Multiplier-sweep sampling (true in-training slider previews).
- Auto-strip an axis-linked phrase list from captions.
- Per-axis coordinate auto-rescale.
- Confounder-stratified sampling (the `stratify` flag is the hook).
- Additional base models beyond Anima.
- Pre-flight cross-field validation surfaced by OT itself.
