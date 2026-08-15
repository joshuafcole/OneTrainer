"""Forensic provenance stamped into sampled images.

A sample image written during training carries no identity beyond its
filename (``{global_step}-{epoch}-{epoch_step}``,
``TrainProgress.filename_string()``) -- fine for the rehearsal scanner, which
lives inside the workspace and already knows that grammar, but useless once
the PNG leaves it: dragged into a chat, or handed to a script running outside
the workspace with no run context.

This module builds the small, flat set of ``ot.``-namespaced keys that answer
"which training state produced this image" without depending on where the file
sits. It imports only the standard library and PIL, so it stays importable --
and unit-testable -- without torch/diffusers on a box with no GPU. Callers
(``BaseModelSampler``, ``GenericTrainer``) own everything that would pull in
the training stack: resolving the actual seed, hashing the resolved
``SampleConfig``, and tracking the last save.

**Both image formats carry it.** PNG gets one ``tEXt`` chunk per field; JPEG
gets the same map as compact JSON in EXIF ``ImageDescription``. JPEG is not the
afterthought here -- it is ``sample_image_format``'s default and what every
real workspace config sets, so a PNG-only implementation would be inert on
exactly the runs this exists for.
"""

import hashlib
import json
from dataclasses import dataclass

from PIL import Image, PngImagePlugin

# EXIF ImageDescription. The free-text tag, chosen over UserComment (0x9286)
# because UserComment mandates an 8-byte character-code prefix that every
# reader then has to strip -- and over a JPEG COM marker because EXIF is what
# exiftool and every image tool surface without being asked.
_EXIF_IMAGE_DESCRIPTION = 0x010E


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
    """Flat, all-string ``ot.``-namespaced key/value map, shared by both formats.

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


def build_exif(prov: SampleProvenance) -> Image.Exif:
    """An ``Exif`` block carrying ``provenance_fields`` as compact JSON in
    ImageDescription, ready to pass as ``Image.save(..., exif=...)``.

    JSON rather than a key=value line so the reader recovers the same dict the
    PNG side gets, verbatim -- one parse, no field-splitting grammar to keep in
    sync between the two formats. ``sort_keys`` makes it byte-stable, which
    matters because two samples of identical training state should produce
    identical provenance bytes."""
    exif = Image.Exif()
    exif[_EXIF_IMAGE_DESCRIPTION] = json.dumps(
        provenance_fields(prov), sort_keys=True, separators=(",", ":")
    )
    return exif


def read_provenance_fields(image: Image.Image) -> dict[str, str]:
    """The ``ot.``-namespaced map back out of an opened image, whichever format
    it was written in -- ``{}`` when the image carries none.

    The inverse of the two builders above, and the reason they exist as a pair:
    a forensic chunk nothing can read is decoration. Non-``ot.`` keys are
    dropped, so an image carrying an unrelated ImageDescription (a caption, a
    tool's watermark) reads as "no provenance" rather than as garbage."""
    text_fields = {
        key: value
        for key, value in image.info.items()
        if isinstance(key, str) and key.startswith("ot.") and isinstance(value, str)
    }
    if text_fields:
        return text_fields

    description = image.getexif().get(_EXIF_IMAGE_DESCRIPTION)
    if not isinstance(description, str):
        return {}
    try:
        decoded = json.loads(description)
    except ValueError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in decoded.items()
        if str(key).startswith("ot.")
    }
