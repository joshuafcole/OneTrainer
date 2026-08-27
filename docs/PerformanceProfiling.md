# Performance profiling (`OT_PERF`)

`OT_PERF=1` turns on a per-step timing probe and writes newline-delimited JSON.
`scripts/analyze_perf.py` turns that log into a per-resolution breakdown and a
power-law fit. Nothing about it is configurable from the UI, and nothing about it
runs unless the environment variable is set.

It exists to answer one class of question: **where does the time go, and how does
that change with resolution?** A trainer that is fine at 512 and unusable at 1024
is not necessarily spending the extra time where you would guess — it may be
attention, or it may be layer offload traffic, or it may be the dataloader, and
those have different fixes.

## Running it

```bash
OT_PERF=1 python scripts/train.py --config-path <your config>
python scripts/analyze_perf.py ot_perf.jsonl
```

| variable | meaning |
|---|---|
| `OT_PERF` | `1` enables the probe. Unset (the default) disables everything below. |
| `OT_PERF_OUT` | Output path. Default `ot_perf.jsonl` in the working directory; appended to, not truncated. |
| `OT_PROFILE_STEP` | Capture one full `torch.profiler` chrome trace at this global step. |
| `OT_PROFILE_MIN_TOKENS` | Capture the trace at the first step, at or after `OT_PROFILE_STEP`, whose latent token count reaches this threshold. |

`OT_PROFILE_MIN_TOKENS` is for runs where you cannot name the step you want in
advance. A VRAM-saturating high-resolution bucket can run at ~0.01 it/s, so
waiting for a fixed late step index is hours, and which step lands in that bucket
depends on shuffling. Setting a threshold instead latches onto the first step big
enough to be interesting; `OT_PROFILE_STEP` then acts as a warmup floor, so the
trace is not just `torch.compile`. Exactly one trace is captured either way.

This is separate from `OT_DEBUG_PROFILES=1`, which captures traces at a fixed set
of step indices and works with or without `OT_PERF`.

## What lands in the log

One `{"kind": "step"}` row per training step:

* `predict_ms`, `prior_predict_ms`, `backward_ms`, `optimizer_ms` — CUDA-event
  timings, resolved with a single synchronize per step.
* `step_total_ms`, `data_wait_ms` — wall clock for the step, and the gap since the
  previous step ended, i.e. time the GPU spent waiting on data or host work.
* `latent_tokens`, `batch_size` — `h_lat * w_lat`, the self-attention sequence
  length. Every other number is only comparable to another row with the same value.
* `vram_peak_alloc_gb`, `vram_peak_reserved_gb`, `recompiles`.
* `offload_xfers`, `offload_onload` — layer transfers actually performed this step
  (present only when layer offloading moved something).
* `rank` — which GPU wrote the row. Under multi-GPU every rank appends to the same
  `OT_PERF_OUT`, so filter on this before reading a breakdown as one device's.

One `{"kind": "cache"}` row per cached `(group, variation)` build, from the mgds
side: items, wall clock, items/s, and megapixels/s of VAE encoding.

## Reading the fit

`analyze_perf.py` buckets step rows by `latent_tokens`, takes medians (dropping the
first row of each bucket, which carries `torch.compile` warmup) and fits
`log(time)` against `log(tokens)`:

```
  power-law fit vs latent_tokens (1.0 = linear in sequence length, ~2.0 = quadratic):
    predict              ~ tokens^1.31
    backward             ~ tokens^1.28
    step_total           ~ tokens^1.24   -> ~ resolution^2.48 per side
```

An exponent near 1.0 on `predict` means attention is behaving linearly in
sequence length, as a fused/flash kernel does. Near 2.0 means it is not, and the
fix is a kernel or backend question rather than a scheduling one. Tokens go as
resolution squared, so the exponent on the last row is doubled to give the one
you feel when you move the resolution slider.

## Cost when it is off

Every call site is guarded with `if perf.enabled:` rather than relying on the
probe to return early, because Python evaluates arguments before the callee can
decline them. With `OT_PERF` unset a hook costs one attribute load — around 6 ns,
against ~180 ns for the same site left unguarded. `tests/test_perf_hooks.py`
measures both and asserts every hook under `modules/` is guarded.
