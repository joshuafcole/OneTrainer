"""Per-model-type hard caps on aspect-bucket resolution.

Single source of truth shared by two consumers that must never disagree:

  - the dataloader (``AnimaBaseDataLoader``), which passes the cap into
    ``AspectBucketing`` so extreme aspect rungs are scaled down instead of
    crashing the model, and
  - the latent cache salt (``modules/util/cache_key.py``), which folds the cap
    into the image-cache identity so a cache produced *before* a cap existed (or
    with a different cap) lands in a fresh directory instead of silently serving
    stale, oversized latents.

If these two read the cap from different places they can drift -- the salt could
say "capped" while the buckets weren't, or vice versa -- which is exactly the
class of silent-stale-cache bug the salt is meant to prevent. So both import from
here. This module stays dependency-light (only ``ModelType``) so the pure cache
salt and its isolated unit test can import it without pulling in the training
stack.
"""

from __future__ import annotations

from modules.util.enum.ModelType import ModelType

# Hard cap on an aspect bucket's longest edge, in pixels, for Anima/Cosmos.
#
# Cosmos's RoPE precomputes a positional table sized from max_size=(128, 240, 240)
# // patch=(1, 2, 2) -> [128, 120, 120] (temporal, height, width in patches). The
# shared position index is seq = arange(max(...)) = arange(128). CosmosRotaryPosEmbed
# then slices seq[:pe_size] per axis, which SILENTLY TRUNCATES once a side's patch
# count exceeds the table -- and the freqs torch.cat downstream blows up on the
# resulting shape mismatch (the symptom: e.g. a 2176-wide bucket from extreme aspect
# rungs at high training resolution).
#
# Patches relate to pixels by vae_scale_factor(8) * patch(2) = 16, so:
#   - 128 patches == 2048 px is the hard crash boundary (the seq table length).
#   - 120 patches == 1920 px is the model's pretrained spatial range (max_size's
#     spatial axes). We cap here rather than at the crash boundary so buckets stay
#     in-distribution and avoid extrapolating RoPE past positions the model ever saw.
# 1920 is a multiple of the bucket quantization (64), so capped edges quantize cleanly.
ANIMA_MAX_BUCKET_RESOLUTION = 1920

# Model types that need a long-edge cap; absence means uncapped (the default for
# every other model). A mapping rather than a chain of conditionals so the table
# of which models are constrained stays readable as more are added.
_MAX_BUCKET_RESOLUTION: dict[ModelType, int] = {
    ModelType.ANIMA: ANIMA_MAX_BUCKET_RESOLUTION,
}


def max_bucket_resolution_for(model_type: ModelType) -> int | None:
    """The aspect-bucket long-edge cap (px) for ``model_type``, or ``None`` if it
    is uncapped. Keyed by the real enum, so a duck-typed/fake model_type that is
    not a genuine ``ModelType`` member returns ``None`` (uncapped) -- the safe
    default that leaves non-capped models' behavior and cache salts unchanged."""
    return _MAX_BUCKET_RESOLUTION.get(model_type)
