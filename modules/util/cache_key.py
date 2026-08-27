"""Content-addressing salts for the on-disk latent / text-embedding cache.

mgds' ``DiskCache`` keys a cached sample only by its per-concept *group key*
(``concept.path``, ``concept.seed``, ``concept.include_subdirectories`` and the
``concept.image`` / ``concept.text`` sub-config), and its only staleness check is
the row count. That is fine while the cache is cleared before every run, but it
silently serves wrong tensors the moment a cache is reused
(``clear_cache_before_training = False``) after a change the group key does not
capture -- the VAE, the training resolution, the aspect-bucket geometry, a text
encoder, or the dataset itself.

These salts capture exactly those *global* (not-per-concept) inputs. The data
loader nests the image/text cache directories under the matching salt, so a
changed identity lands in a fresh directory instead of colliding with stale
tensors -- and an unchanged identity reuses the cache for free. The salt and the
group key compose: the salt segregates the global dimensions, the group key the
per-concept ones.

Three kinds of input go in.

**Model identity** is path/string based on purpose: hashing multi-GB checkpoint
weights at every launch would defeat the point of a cache. The gap that leaves --
swapping different weights in at the *same* path -- is the documented cost of
trading weight-hashing for launch speed.

**The cached item's shape**: the split/aggregate names the ``DiskCache`` is built
with. This is not cosmetic. ``DiskCache.get_item`` does an unguarded
``item[name] = aggregate_item[name]`` over its ``aggregate_names``, and the
staleness check cannot see a same-length change, so a run whose name set grew
against an existing cache raises ``KeyError`` from inside the dataloader rather
than re-caching. The name sets are derived from config (masked training, custom
conditioning images, whether the bucket planner emits keep/repeat tags), so the
salt has to carry them or the user pays for a setting they toggled with a crash.
Keying on the names themselves rather than on the settings that produce them
means a future name is covered without anyone remembering to come back here.

**The bucket geometry**, taken from the one ``BucketingParams`` both bucketing
modules are built from rather than re-derived from config. Re-derivation would
have to restate the per-dataloader quantization and long-edge cap, which are
constructor arguments and not config fields, and could then drift from what the
pipeline actually used.

**The dataset**, via ``modules/util/dataset_key.py`` -- see that module for why
media and captions are fingerprinted separately.

One cost worth stating: a salt change abandons the previous directory rather than
overwriting it, so a workspace whose configuration moves around accumulates cache
directories. ``clear_cache_before_training`` removes them all; nothing prunes
individual stale salts.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from modules.util.bucket_tiers import BucketingParams
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.dataset_key import dataset_fingerprints


@dataclass(frozen=True)
class CacheSalts:
    """The directory name each cache nests under."""

    image: str
    text: str


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _included_part_name(part: TrainModelPartConfig) -> str | None:
    """The checkpoint name of a model part if it is actually used, else ``None``."""
    if not part.include:
        return None
    return part.model_name or ""


def _bucket_identity(config: TrainConfig, bucketing: BucketingParams) -> dict[str, Any] | None:
    """The bucket geometry that decides an image's crop, or ``None`` when bucketing
    is off and none of it reaches the pipeline.

    ``batch_size`` is folded in only when a tier exists. It is an argument to the
    rebalancing planner and to nothing else, and ``plan_rebalance`` returns a no-op
    plan without tiers -- so including it unconditionally would re-encode every
    latent whenever a user changed their batch size, for no change in any tensor.
    """
    if not config.aspect_ratio_bucketing:
        return None
    return {
        "quantization": bucketing.quantization,
        "tolerance": bucketing.tolerance,
        "resolution_mode": bucketing.resolution_mode,
        "max_resolution": bucketing.max_resolution,
        # Sorted, and taken from the *parsed* tiers: the planner sorts them by
        # max_size anyway, and parsing drops the blank rows the tier editor leaves
        # behind. Neither reordering a tier list nor leaving an unfilled row in it
        # changes a single crop, so neither may re-encode a dataset.
        "tiers": sorted((tier.max_size, tier.strategy, tier.mode) for tier in bucketing.tiers),
        "batch_size": bucketing.batch_size if bucketing.tiers else None,
    }


def cache_salts(
        config: TrainConfig,
        bucketing: BucketingParams,
        concepts: list[ConceptConfig],
        image_names: list[str],
        text_names: list[str],
) -> CacheSalts:
    """Both cache salts for one dataset.

    ``image_names`` / ``text_names`` are the names the corresponding ``DiskCache``
    is built with (split and aggregate together); ``concepts`` are the concepts
    that actually feed this dataset, already resolved from the concept file and
    filtered for validation.

    Computed together because the two share one walk of the dataset, which is the
    expensive part.
    """
    media_fingerprint, caption_fingerprint = dataset_fingerprints(concepts)
    return CacheSalts(
        image=_image_salt(config, bucketing, image_names, media_fingerprint),
        text=_text_salt(config, text_names, caption_fingerprint),
    )


def _image_salt(
        config: TrainConfig,
        bucketing: BucketingParams,
        image_names: list[str],
        media_fingerprint: str,
) -> str:
    """Identity of everything that changes a cached VAE latent for a given source
    image and crop.

    Per-concept image settings (crop jitter, flip, a per-concept resolution
    override) already live in the DiskCache group key via ``concept.image`` and
    deliberately do **not** belong here.

    Captions are deliberately excluded: they change no VAE output, and folding them
    in would push a whole dataset back through the VAE because one word was
    reworded.
    """
    return _digest({
        "model_type": str(config.model_type),
        # An empty vae.model_name means "use the base model's VAE", so fall back to it.
        "vae": config.vae.model_name or config.base_model_name,
        "resolution": config.resolution,
        "frames": config.frames,
        "bucketing": _bucket_identity(config, bucketing),
        "names": sorted(image_names),
        "dataset_media": media_fingerprint,
    })


def _text_salt(config: TrainConfig, text_names: list[str], caption_fingerprint: str) -> str:
    """Identity of everything that changes a cached text-encoder embedding for a
    given prompt: each *included* text encoder (checkpoint, layer skip, sequence
    length) and any embedding that extends the tokenizer vocabulary.

    Per-concept caption settings already live in the group key via ``concept.text``.

    Embeddings are folded in conservatively: whether an added token actually shifts
    a model's cached hidden states is model-specific, so this over-invalidates (a
    spurious re-cache) rather than risk reusing embeddings computed without it.

    Bucketing is absent on purpose. It reaches no text encoder: the only way it
    touches the text pipeline at all is borrow-copy, which appends duplicate rows,
    and appended rows change the row count -- which is precisely what the
    DiskCache's own length check catches and rebuilds on.
    """
    # (part, layer skip, sequence length). Only encoder 2 has a configurable
    # sequence length today; the fields are read straight off TrainConfig rather
    # than through getattr so a renamed or removed one is a failure here instead of
    # a silently constant contribution to the digest.
    specs: list[tuple[TrainModelPartConfig, int, int | None]] = [
        (config.text_encoder, config.text_encoder_layer_skip, None),
        (config.text_encoder_2, config.text_encoder_2_layer_skip, config.text_encoder_2_sequence_length),
        (config.text_encoder_3, config.text_encoder_3_layer_skip, None),
        (config.text_encoder_4, config.text_encoder_4_layer_skip, None),
    ]
    encoders = [
        {"name": name, "layer_skip": layer_skip, "seq_len": seq_len}
        for part, layer_skip, seq_len in specs
        if (name := _included_part_name(part)) is not None
    ]

    embeddings = [
        {
            "name": embedding.model_name or "",
            "placeholder": embedding.placeholder or "",
            "token_count": embedding.token_count,
        }
        for embedding in [config.embedding, *config.additional_embeddings]
        if embedding.model_name or embedding.placeholder
    ]

    return _digest({
        "model_type": str(config.model_type),
        "encoders": encoders,
        "embeddings": embeddings,
        "names": sorted(text_names),
        # The captions behind the embeddings -- plus the media file *list*, since
        # embeddings are cached positionally per row and which rows exist is part of
        # the identity even though their pixels are not. Media size/mtime are
        # excluded, so re-encoding an image never re-runs the text encoders.
        "dataset_captions": caption_fingerprint,
    })
