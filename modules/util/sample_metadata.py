"""Forensic provenance stamped into sampled images.

A sample image written during training carries no identity beyond its
filename (``{global_step}-{epoch}-{epoch_step}``,
``TrainProgress.filename_string()``) -- fine for the rehearsal scanner, which
lives inside the workspace and already knows that grammar, but useless once
the PNG leaves it: dragged into a chat, or handed to a script running outside
the workspace with no run context.

This module builds the small, flat set of ``ot.``-namespaced PNG text-chunk
keys that answer "which training state produced this image" without
depending on where the file sits. It imports only the standard library and
PIL, so it stays importable -- and unit-testable -- without torch/diffusers on
a box with no GPU. Callers (``BaseModelSampler``, ``GenericTrainer``) own
everything that would pull in the training stack: resolving the actual seed,
hashing the resolved ``SampleConfig``, and tracking the last save.
"""

import hashlib
from dataclasses import dataclass

from PIL import PngImagePlugin


def hash_text(text: str) -> str:
    """sha256 of ``text``, truncated to 16 hex chars -- the same truncation
    ``modules/util/cache_key.py`` uses for its cache-identity salts, so a
    provenance hash reads the same shape as the cache hashes elsewhere in the
    repo."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class SampleProvenance:
    global_step: int
    epoch: int
    epoch_step: int
    seed: int | None
    prompt_hash: str
    sample_config_hash: str
    last_save_filename: str | None


def provenance_fields(prov: SampleProvenance) -> dict[str, str]:
    """Flat, all-string ``ot.``-namespaced key/value map for a PNG text chunk.

    ``None`` fields (an unresolved random seed, no save emitted yet this run)
    are omitted rather than written as the literal string ``"None"`` --
    absence here means "not known", not "known to be empty"."""
    fields = {
        "ot.global_step": str(prov.global_step),
        "ot.epoch": str(prov.epoch),
        "ot.epoch_step": str(prov.epoch_step),
        "ot.prompt_hash": prov.prompt_hash,
        "ot.sample_config_hash": prov.sample_config_hash,
    }
    if prov.seed is not None:
        fields["ot.seed"] = str(prov.seed)
    if prov.last_save_filename is not None:
        fields["ot.last_save_filename"] = prov.last_save_filename
    return fields


def build_png_info(prov: SampleProvenance) -> PngImagePlugin.PngInfo:
    """A ``PngInfo`` block carrying ``provenance_fields`` as tEXt chunks, ready
    to pass as ``Image.save(..., pnginfo=...)``."""
    info = PngImagePlugin.PngInfo()
    for key, value in provenance_fields(prov).items():
        info.add_text(key, value)
    return info
