"""Bring an on-disk LoRA into OneTrainer's canonical (diffusers) key namespace.

**Why this exists.** Which naming an adapter file carries is a *choice*, not a
constant. ``TrainConfig.output_model_format`` selects it, and the LoRA formats
disagree about both halves of a key:

===================  ==================================  ====================
``ModelFormat``      module path                         value suffix
===================  ==================================  ====================
``DIFFUSERS_LORA``   ``transformer.transformer_blocks.``  ``lora_A``/``lora_B``
``COMFY_LORA``       ``diffusion_model.blocks.``          ``lora_A``/``lora_B``
``ORIGINAL_LORA``    bare native (``net.blocks.``)        ``lora_A``/``lora_B``
``KOHYA_LORA``       ``lora_unet_`` flattened             ``lora_down``/``lora_up``
``INTERNAL``         canonical, unconverted               ``lora_down``/``lora_up``
===================  ==================================  ====================

This analysis family identifies a layer *by its key prefix*: ``lora_soup``
splits on the value suffix, and everything above it (``block_gram``,
``block_subspace``, ``block_kron_spectrum``, ``block_compare``,
``block_summary``, ``block_contribution``) compares, intersects and groups those
prefixes across files. So a namespace difference is not a cosmetic one. Two
saves **of the very same run** in two formats share no layer at all, and the
report is ``block_gram``'s "no layers common to all adapters -- these do not
target the same model" -- an accusation about the *model* for what is only a
disagreement about spelling. A LoRA in an A/B-suffix format fares no better: it
reaches ``load_lora`` as a file of entirely unrecognised keys.

Neither is hypothetical. It is what happens the first time an adapter trained
before a format change is compared against one trained after it.

**What it does.** The same two layers OneTrainer's own LoRA *loader* uses
(``modules/util/load_lora_util.py``), driven from a file instead of a live
model:

1. ``normalize_various`` -- the model-independent pre-pass. Peft containers,
   ``lora_A``/``lora_B`` -> ``lora_down``/``lora_up``, the DoRA magnitude
   spellings, and a folded alpha synthesised back to ``alpha = rank``.
2. the namespace reverse, for the one family that needs it here (COMFY), taken
   from the model's own forward table (``model.lora_diffusers_to_comfy()``) so
   it cannot drift from what the saver wrote.

The target is the canonical (diffusers) namespace and not the native one
because that is the namespace ``block_groups.json`` is written in --
``block_prefix: transformer.transformer_blocks``, ``attn1``, ``ff.net``. Canonicalising
here leaves the taxonomy untouched.

**What it does not do.** ``KOHYA_LORA``, ``LEGACY_LORA`` and ``ORIGINAL_LORA``
are refused, not converted. Un-flattening a kohya path is ambiguous by rule and
is resolved upstream against the *live* model's module names
(``kohya_unflatten``), which an offline script reading one safetensors file does
not have. Refusing them is a decision: a wrong un-flatten produces layer names
that look right and group wrong, and a silently mis-grouped ablation is worse
than one that did not run.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# Reach ``modules/`` the way lora_soup does -- repo root at the head of sys.path, no ZLUDA half -- so this
# is importable both as a library of the block_* scripts and on its own.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.util.convert_lora_util import (  # noqa: E402
    convert_lora_suffix_ab,
    lora_original_conversion,
)
from modules.util.convert_util import convert  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.enum.ModelType import ModelType  # noqa: E402
from modules.util.load_lora_util import (  # noqa: E402
    normalize_key_names,
    normalize_various,
    reverse_comfy,
)

from torch import Tensor  # noqa: E402

#: COMFY's denoising top prefix (``LoRASaverMixin._save_comfy``).
COMFY_PREFIX = "diffusion_model."

#: KOHYA's flattened top prefixes (``lora_kohya_conversion`` + ``kohya_flatten``).
KOHYA_PREFIXES: tuple[str, ...] = ("lora_unet_", "lora_te")

#: Top-level key segments that are not a model component and belong to no namespace.
PASSTHROUGH_SEGMENTS: frozenset[str] = frozenset({"bundle_emb"})

#: ``modelspec.architecture`` stem -> the model that owns the namespace tables.
#: The tables themselves are never copied here -- only how to reach the class
#: that declares them -- so this cannot fall out of step with a saver. One entry
#: per model whose adapters this analysis family is run against; anything else
#: gets :class:`NamespaceError` rather than a guess. Imported lazily: resolving a
#: model pulls in diffusers/transformers, which a file already in canonical
#: naming has no reason to pay for.
ARCHITECTURE_MODELS: dict[str, tuple[ModelType, str, str]] = {
    "anima": (ModelType.ANIMA, "modules.model.AnimaModel", "AnimaModel"),
}


class NamespaceError(Exception):
    """The file's key namespace cannot be brought to canonical."""


def _model_parts() -> frozenset[str]:
    """Every component name any model declares, plus the passthrough segments.

    The canonical namespace prefixes each key with its component
    (``transformer.``, ``unet.``, ``prior.``, ``text_encoder.``), so this is the
    admissible first segment of a canonical key -- and a file whose first
    segments are outside it is in some namespace we have not identified, which
    is a thing to say rather than to average over.
    """
    parts = set(PASSTHROUGH_SEGMENTS)
    for model_type in ModelType:
        parts.update(model_type.model_parts())
    return frozenset(parts)


def family(keys: Iterable[str]) -> str:
    """Which on-disk namespace ``keys`` are in: ``canonical``, ``comfy``,
    ``kohya_flat`` or ``foreign``.

    Structural and model-free, so the common case -- a file already canonical --
    is decided without importing a model. ``foreign`` is the honest answer for
    ORIGINAL/LEGACY and for anything unrecognised: they are distinguishable from
    each other only against a live model, and this reports what it knows.
    """
    keys = list(keys)
    if any(key.startswith(COMFY_PREFIX) for key in keys):
        return "comfy"
    if any(key.startswith(KOHYA_PREFIXES) for key in keys):
        return "kohya_flat"
    parts = _model_parts()
    if all(key.split(".", 1)[0] in parts for key in keys):
        return "canonical"
    return "foreign"


class _DeclaredTextEncoder:
    """Presence marker for a text encoder this script has no weights for.

    ``lora_text_encoders()`` is documented as the model's *only* LoRA-namespace
    declaration, but every implementation gates its entries on the live module
    being loaded -- ``StableDiffusion3Model``'s says so outright ("any can be
    absent, so only the TEs actually present are declared"). That is the right
    rule for a loader, which needs the module to read its parameter names from,
    and the wrong one here: :func:`_model_for` builds the model with every field
    ``None``, so a weightless model declares **no** text encoders and the
    namespace tables come back short.

    Short in a way that matters. ``lora_original_conversion`` appends the
    ``bundle_emb`` passthrough *only when the model declares a text encoder*
    (bundled embeddings live in that namespace), and runs its ``convert`` with
    ``strict=True``. So a LoRA carrying a bundled TI vector died here with
    ``No conversion found for key bundle_emb.<placeholder>.qwen`` -- a file the
    saver had just written, refused by the saver's own table read one field
    short.

    Offline, presence is the file's to decide, not a loaded model's: a
    ``bundle_emb.`` key is in the file precisely because a saver with a live
    encoder put it there. So the gates are opened, the declaration is read
    whole, and the key set decides which parts of it fire (an overdefined
    conversion whose source component is absent never does).

    Deliberately not an ``nn.Module``: nothing reads this object today --
    ``lora_text_encoders`` hands the module straight through and only the name
    dict is used -- and if some future declaration does read it, an
    ``AttributeError`` naming the attribute is a better answer than a silently
    empty text-encoder list.
    """

    def __repr__(self) -> str:
        return "<text encoder declared, weights not loaded>"


#: Instance attributes that hold a text encoder, by the naming every model uses
#: (``text_encoder``, ``text_encoder_1`` ...). Anchored, so the neighbours that
#: merely start the same way -- ``text_encoder_embedding``,
#: ``text_encoder_train_dtype``, ``text_encoder_offload_conductor`` -- are left
#: alone.
_TEXT_ENCODER_ATTR = re.compile(r"text_encoder(_\d+)?$")


def _declare_text_encoders(model):
    """Open ``model``'s text-encoder declaration gates. See :class:`_DeclaredTextEncoder`."""
    for name in list(vars(model)):
        if _TEXT_ENCODER_ATTR.fullmatch(name) and getattr(model, name) is None:
            setattr(model, name, _DeclaredTextEncoder())
    return model


def _model_for(header: dict[str, str], source: Path | str):
    """The model whose namespace tables describe ``source``, built without weights.

    Identified from the file itself, in the order of decreasing authority: the
    ``ot_config`` block a OneTrainer save carries when
    ``include_train_config`` is on (it names the ``model_type`` outright), then
    the sai ``modelspec.architecture``. The model is constructed with every
    field ``None`` -- the conversion tables are declarations and read no
    weights -- and then handed to :func:`_declare_text_encoders`, because one of
    them does read a field: ``lora_text_encoders`` gates on the live module.
    """
    config = header.get("ot_config")
    if config is not None:
        try:
            model_type_name = json.loads(config).get("model_type")
        except json.JSONDecodeError:
            model_type_name = None
        if model_type_name is not None:
            for model_type, module_name, class_name in ARCHITECTURE_MODELS.values():
                if model_type.name == model_type_name:
                    return _declare_text_encoders(
                        getattr(importlib.import_module(module_name), class_name)(model_type)
                    )

    architecture = header.get("modelspec.architecture", "")
    stem = architecture.split("/", 1)[0].strip().lower()
    entry = ARCHITECTURE_MODELS.get(stem)
    if entry is None:
        known = ", ".join(sorted(ARCHITECTURE_MODELS)) or "<none>"
        raise NamespaceError(
            f"{source}: this adapter is not in the canonical key namespace, and its header does not say "
            f"which model it was trained for, so its names cannot be translated "
            f"(modelspec.architecture={architecture!r}). Known: {known}."
        )
    model_type, module_name, class_name = entry
    return _declare_text_encoders(getattr(importlib.import_module(module_name), class_name)(model_type))


def _comfy_arguments(model) -> tuple[str, list | None, list[str], dict[str, str]]:
    """``reverse_comfy``'s per-model arguments, read off the model's declarations.

    The denoising body is the very list ``_save_comfy`` applied forward; the
    text-encoder prefixes come from ``lora_text_encoders()``, whose live-module
    half is deliberately not touched -- only the names are wanted. It is a
    :class:`_DeclaredTextEncoder` here rather than a real encoder, which is what
    makes the declaration readable at all without weights.
    """
    text_encoders = model.lora_text_encoders()
    return (
        model.model_type.denoising_model_part(),
        model.lora_diffusers_to_comfy(),
        [names[ModelFormat.DIFFUSERS_LORA] for _module, names in text_encoders],
        {names[ModelFormat.DIFFUSERS_LORA]: names[ModelFormat.COMFY_LORA]
         for _module, names in text_encoders if ModelFormat.COMFY_LORA in names},
    )


def _refuse(found: str, source: Path | str) -> NamespaceError:
    return NamespaceError(
        f"{source}: this adapter is in the {found} key namespace, which this analysis family cannot "
        "translate to the canonical one it compares layers in (un-flattening a kohya path, or telling a "
        "bare-native file from a prefix-stripped canonical one, is resolved against the live model's "
        "module names, which reading one safetensors file does not have). Re-save it with "
        "`output_model_format: COMFY_LORA` or `DIFFUSERS_LORA` -- OneTrainer's own "
        "`scripts/convert_model.py --output-model-format COMFY_LORA` rewrites an existing file."
    )


def canonicalize(
        state_dict: dict[str, Tensor],
        header: dict[str, str],
        source: Path | str,
) -> dict[str, Tensor]:
    """``state_dict`` in the canonical (diffusers) namespace, whatever it arrived in.

    An already-canonical file still goes through layer 1 -- a DIFFUSERS save is
    canonical in its *names* and A/B in its *suffixes*, and its alpha was folded
    away, so "already canonical" is a statement about one dimension only.
    """
    state_dict = normalize_various(state_dict)
    found = family(state_dict)
    if found == "canonical":
        return state_dict
    if found == "comfy":
        return reverse_comfy(state_dict, *_comfy_arguments(_model_for(header, source)))
    raise _refuse("kohya-flattened" if found == "kohya_flat" else "an unrecognized", source)


def nativize(
        state_dict: dict[str, Tensor],
        header: dict[str, str],
        source: Path | str,
) -> dict[str, Tensor]:
    """``state_dict`` in the COMFY namespace, whatever it arrived in.

    :func:`canonicalize`'s mirror image, and the reason it can be written at all
    is that the tables are declarations: ``lora_diffusers_to_comfy`` and
    ``lora_text_encoders`` are the same two the saver reads, so running them
    forward here reproduces ``LoRASaverMixin._save_comfy`` rather than
    re-deriving it. A conversion that re-derived the table would be a second
    naming authority, and the first thing it would do is drift.

    This exists because a save in DIFFUSERS naming is not loadable by ComfyUI,
    and until now the only thing that could fix one was a fork-only script that
    upstream #1563 deleted when it made the output format selectable. The
    documented replacement -- ``scripts/convert_model.py --output-model-format
    COMFY_LORA`` -- loads the *base model* to do it, which for a pure key rename
    means finding a base path on the box, several GB of I/O and a GPU. The names
    are the whole job; nothing here reads a weight.

    Refuses on the same condition the saver refuses on: a trained text encoder
    with no Comfy-native name would be **silently dropped** by ComfyUI on load,
    and half a LoRA that loads without error is worse than one that does not.
    """
    state_dict = canonicalize(state_dict, header, source)
    model = _model_for(header, source)
    component = model.model_type.denoising_model_part()

    state_dict = convert(
        state_dict, lora_original_conversion(model, model.lora_diffusers_to_comfy()), strict=True
    )
    te_prefixes = {
        names[ModelFormat.DIFFUSERS_LORA]: names[ModelFormat.COMFY_LORA]
        for _module, names in model.lora_text_encoders()
        if ModelFormat.COMFY_LORA in names
    }
    present = {key.split(".", 1)[0] for key in state_dict} - {component} - PASSTHROUGH_SEGMENTS
    missing = present - te_prefixes.keys()
    if missing:
        raise NamespaceError(
            f"{source}: the COMFY namespace has no Comfy-native text-encoder name for "
            f"{', '.join(sorted(missing))} on this model, so ComfyUI would drop those keys on "
            "load without saying so. Refusing to write a half-readable adapter."
        )
    # The denoising component takes Comfy's prefix; the text encoders take their
    # own; ``bundle_emb.`` passes through. strict=False on both, for that reason.
    state_dict = convert(state_dict, [(component, COMFY_PREFIX.rstrip("."))], strict=False)
    if te_prefixes:
        state_dict = convert(state_dict, list(te_prefixes.items()), strict=False)
    # COMFY's value suffix is lora_A/lora_B, and it honours alpha and dora_scale.
    return convert_lora_suffix_ab(state_dict, peft_convention=False)


def canonicalize_keys(
        keys: Iterable[str],
        header: dict[str, str],
        source: Path | str,
) -> list[str]:
    """:func:`canonicalize` for a caller that has key names and no tensors.

    Runs the rename passes only. The two it skips do not move a key: the DoRA
    spelling collapse renames a value suffix no layer prefix is taken from, and
    the alpha synthesis adds an ``.alpha`` to a module that already named itself
    through its ``lora_up``. So the *prefix* set this returns is the one
    :func:`canonicalize` would produce, which is all a taxonomy reads.
    """
    state_dict = normalize_key_names(dict.fromkeys(keys))
    found = family(state_dict)
    if found == "canonical":
        return sorted(state_dict)
    if found == "comfy":
        return sorted(reverse_comfy(state_dict, *_comfy_arguments(_model_for(header, source))))
    raise _refuse("kohya-flattened" if found == "kohya_flat" else "an unrecognized", source)
