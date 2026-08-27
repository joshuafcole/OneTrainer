"""Per-model-type hard caps on aspect-bucket resolution.

Aspect bucketing gives every rung the same pixel budget, so an extreme aspect
buys its width by getting very wide. Some models cannot accept that: their
positional embeddings are precomputed for a bounded number of patches per side,
and a bucket past that limit either truncates silently or crashes. A cap scales
such buckets down with their aspect preserved, trading the requested pixel budget
for one the model can actually take.

Kept in its own module, with only ``ModelType`` as a dependency, so anything that
needs to know a model's cap can read it from one place instead of restating the
number.
"""

from modules.util.enum.ModelType import ModelType

# Hard cap on an aspect bucket's longest edge, in pixels, for Anima/Cosmos.
#
# Cosmos's RoPE precomputes a positional table from max_size=(128, 240, 240) //
# patch=(1, 2, 2) -> [128, 120, 120] patches (temporal, height, width). The shared
# position index is arange(max(...)) = arange(128), and CosmosRotaryPosEmbed slices
# it per axis, which silently truncates once a side's patch count passes the table;
# the freqs concatenation downstream then fails on the resulting shape mismatch.
#
# vae_scale_factor(8) * patch(2) = 16 pixels per patch, so:
#   - 128 patches == 2048 px is the crash boundary (the length of the table).
#   - 120 patches == 1920 px is the pretrained spatial range (max_size's spatial
#     axes). Capping at the pretrained range rather than the crash boundary keeps
#     buckets in distribution instead of extrapolating RoPE past anything the model
#     was ever shown.
# 1920 is a multiple of every bucket quantization used here (Anima's is 64), which
# the cap must be: the cap is applied before the crop is quantized, and quantizing
# an off-grid cap can round the edge back above it.
ANIMA_MAX_BUCKET_RESOLUTION = 1920

# Model types that need a long-edge cap. Absence means uncapped, which is the case
# for every other model; a table rather than a chain of conditionals so the set of
# constrained models stays readable as it grows.
_MAX_BUCKET_RESOLUTION: dict[ModelType, int] = {
    ModelType.ANIMA: ANIMA_MAX_BUCKET_RESOLUTION,
}


def max_bucket_resolution_for(model_type: ModelType) -> int | None:
    """The aspect-bucket long-edge cap in pixels for ``model_type``, or None if it is
    uncapped. Keyed on the real enum, so anything that is not a genuine ``ModelType``
    member reads as uncapped -- the default that leaves every other model unchanged."""
    return _MAX_BUCKET_RESOLUTION.get(model_type)
