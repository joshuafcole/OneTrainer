"""Content-addressing salts for the on-disk latent / text-embedding cache.

The MGDS ``DiskCache`` keys a cached sample only by its per-concept *group key*
(``concept.path``, ``concept.seed``, ``concept.include_subdirectories`` and the
``concept.image`` / ``concept.text`` sub-config) and does **no** staleness check
beyond "does ``aggregate.pt`` exist". That is fine while the cache is cleared
before every run, but it silently serves wrong tensors the moment you reuse a
cache (``clear_cache_before_training = False``) after changing something the group
key doesn't capture — the VAE, the training resolution / aspect bucketing, or a
text encoder.

These two salts capture exactly those *global* (not-per-concept) inputs. The data
loader nests the image/text cache directories under the matching salt, so a
changed VAE/resolution/encoder lands in a fresh directory instead of colliding
with stale tensors — and an unchanged identity reuses the cache for free. The salt
and the DiskCache group key compose: salt segregates the global dimensions, the
group key segregates the per-concept ones.

They also fold in the *dataset* — the contents of every enabled concept dir, via
``modules/util/dataset_key.py``. The group key holds a concept's ``path`` but never
its contents, so without this an edited dataset at an unchanged path reuses the cache
outright. That is not hypothetical: a staging layer that reuses one path across
re-exports (to get incremental blob sync) produces exactly that shape, and reusing
the cache under it served a caption's embedding for a different caption, and — when
the file count moved too — indexed off the end of the cached list entirely. Media and
captions are fingerprinted separately so a caption edit does not re-encode the VAE.

Identity is path/string based on purpose: hashing multi-GB checkpoint weights at
every launch would defeat the whole point. The one gap that leaves — swapping a
different file in at the *same* model path — is rare for checkpoints and is the
documented limitation of trading weight-hashing for launch speed. Dataset files take
the same trade for media (size + mtime) but not for captions, which are small enough
to hash outright; see ``dataset_key`` for why.
"""

import hashlib
import json
from typing import Any

from modules.util.bucket_limits import max_bucket_resolution_for
from modules.util.dataset_key import dataset_fingerprints


def _digest(payload: Any) -> str:
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _included_part_name(part: Any) -> str | None:
    """The checkpoint name of a ``TrainModelPartConfig`` if it's actually used."""
    if part is None:
        return None
    if not getattr(part, "include", True):
        return None
    return getattr(part, "model_name", "") or ""


def image_cache_salt(config) -> str:
    """Identity of everything that changes a cached VAE latent for a given source
    image + crop. Per-concept image settings (crop jitter, flip, per-concept
    resolution override) already live in the DiskCache group key via
    ``concept.image`` and deliberately do **not** belong here."""
    vae = getattr(config, "vae", None)
    vae_name = (getattr(vae, "model_name", "") or "") if vae is not None else ""
    payload = {
        "v": 1,
        "model_type": str(config.model_type),
        # Empty vae.model_name means "use the base model's VAE", so fall back to it.
        "vae": vae_name or config.base_model_name,
        "resolution": config.resolution,
        "frames": getattr(config, "frames", None),
        "bucketing": bool(config.aspect_ratio_bucketing),
        "bucket_tolerance": config.aspect_ratio_bucket_tolerance,
        "bucket_resolution_mode": getattr(config, "aspect_ratio_bucket_resolution_mode", None),
        "bucket_min_tiers": config.aspect_ratio_bucket_min_tiers or [],
    }
    # A model-type long-edge cap changes the crop resolution of extreme aspect
    # rungs, so a cache built before the cap existed (or with a different cap)
    # must not be reused. Added only when a cap applies: omitting the key for
    # uncapped models keeps their salt byte-identical to pre-cap builds, so they
    # don't pay a spurious one-time re-cache.
    bucket_max_resolution = max_bucket_resolution_for(config.model_type)
    if bucket_max_resolution is not None:
        payload["bucket_max_resolution"] = bucket_max_resolution
    # The media behind the latents. Captions are deliberately excluded: they change
    # no VAE output, and folding them in here would push a whole dataset back through
    # the VAE because one word was reworded.
    media_fingerprint, _ = dataset_fingerprints(getattr(config, "concepts", None))
    if media_fingerprint is not None:
        payload["dataset_media"] = media_fingerprint
    return _digest(payload)


def text_cache_salt(config) -> str:
    """Identity of everything that changes a cached text-encoder embedding for a
    given prompt: each *included* text encoder (checkpoint + layer skip + sequence
    length) and any embeddings that extend the tokenizer vocabulary. Per-concept
    caption settings already live in the group key via ``concept.text``.

    Embeddings are folded in conservatively: whether an added token actually shifts
    a model's cached hidden states is model-specific, so we over-invalidate (a
    spurious re-cache) rather than risk reusing embeddings computed without it."""
    encoders = []
    # (part attr, layer-skip attr, sequence-length attr)
    specs = [
        ("text_encoder", "text_encoder_layer_skip", None),
        ("text_encoder_2", "text_encoder_2_layer_skip", "text_encoder_2_sequence_length"),
        ("text_encoder_3", "text_encoder_3_layer_skip", "text_encoder_3_sequence_length"),
        ("text_encoder_4", "text_encoder_4_layer_skip", "text_encoder_4_sequence_length"),
    ]
    for part_attr, skip_attr, seq_attr in specs:
        name = _included_part_name(getattr(config, part_attr, None))
        if name is None:
            continue
        encoders.append(
            {
                "name": name,
                "layer_skip": getattr(config, skip_attr, None),
                "seq_len": getattr(config, seq_attr, None) if seq_attr else None,
            }
        )

    embeddings = []
    primary = getattr(config, "embedding", None)
    candidates = ([primary] if primary is not None else []) + list(
        getattr(config, "additional_embeddings", None) or []
    )
    for emb in candidates:
        name = getattr(emb, "model_name", "") or ""
        placeholder = getattr(emb, "placeholder", "") or ""
        if name or placeholder:
            embeddings.append({"name": name, "placeholder": placeholder})

    payload = {
        "v": 1,
        "model_type": str(config.model_type),
        "encoders": encoders,
        "embeddings": embeddings,
    }
    # The captions behind the embeddings — plus the media file *list*, since text
    # embeddings are cached positionally per row and which rows exist is part of the
    # identity even though their pixels are not. Media size/mtime are excluded, so
    # re-encoding an image never re-runs the text encoders.
    _, caption_fingerprint = dataset_fingerprints(getattr(config, "concepts", None))
    if caption_fingerprint is not None:
        payload["dataset_captions"] = caption_fingerprint
    return _digest(payload)
