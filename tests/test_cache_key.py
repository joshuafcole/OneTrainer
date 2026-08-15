"""Behavioural tests for the latent/text cache salts (modules/util/cache_key.py).

The helpers are pure (stdlib only, duck-typed config), so these run without the
training stack: ``python tests/test_cache_key.py`` or under pytest. They guard the
property that makes reusing a non-cleared cache safe — a salt must move when (and
only when) an input that changes the cached tensors changes.
"""

import importlib.util
import os
import sys
import tempfile
import types

# cache_key now imports modules.util.bucket_limits, so the repo root must be on the
# path for its isolated file-location load to resolve (repo test convention).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.util import dataset_key
from modules.util.bucket_limits import ANIMA_MAX_BUCKET_RESOLUTION, max_bucket_resolution_for
from modules.util.enum.ModelType import ModelType

_spec = importlib.util.spec_from_file_location(
    "cache_key",
    os.path.join(os.path.dirname(__file__), "..", "modules", "util", "cache_key.py"),
)
cache_key = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_key)
image_cache_salt = cache_key.image_cache_salt
text_cache_salt = cache_key.text_cache_salt


class _Part:
    def __init__(self, model_name="", include=True):
        self.model_name = model_name
        self.include = include


class _Emb:
    def __init__(self, model_name="", placeholder=""):
        self.model_name = model_name
        self.placeholder = placeholder


class _ModelType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"ModelType.{self.name}"


def _cfg(**overrides):
    base = {
        "model_type": _ModelType("STABLE_DIFFUSION_XL_10_BASE"),
        "base_model_name": "/models/sdxl.safetensors",
        "vae": _Part(""),  # empty -> falls back to base_model_name
        "resolution": "1024",
        "frames": "25",
        "aspect_ratio_bucketing": True,
        "aspect_ratio_bucket_tolerance": 0.0,
        "aspect_ratio_bucket_resolution_mode": "split",
        "aspect_ratio_bucket_min_tiers": [],
        "text_encoder": _Part("clip-l"),
        "text_encoder_layer_skip": 0,
        "text_encoder_2": _Part("clip-g"),
        "text_encoder_2_layer_skip": 0,
        "text_encoder_2_sequence_length": 77,
        "text_encoder_3": _Part("", include=False),
        "text_encoder_4": _Part("", include=False),
        "embedding": _Emb(),
        "additional_embeddings": [],
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_salts_are_deterministic_16_hex():
    salt = image_cache_salt(_cfg())
    assert salt == image_cache_salt(_cfg())
    assert text_cache_salt(_cfg()) == text_cache_salt(_cfg())
    assert len(salt) == 16 and all(c in "0123456789abcdef" for c in salt)


def test_image_salt_tracks_vae_resolution_bucketing():
    assert image_cache_salt(_cfg()) != image_cache_salt(_cfg(resolution="768"))
    assert image_cache_salt(_cfg()) != image_cache_salt(_cfg(vae=_Part("/fix-vae.safetensors")))
    assert image_cache_salt(_cfg()) != image_cache_salt(_cfg(aspect_ratio_bucketing=False))
    # empty VAE means "use the base model's VAE", so the base model must count
    assert image_cache_salt(_cfg()) != image_cache_salt(_cfg(base_model_name="/other.safetensors"))


def test_image_salt_ignores_text_only_changes():
    assert image_cache_salt(_cfg()) == image_cache_salt(_cfg(text_encoder=_Part("other")))


def test_bucket_cap_only_for_capped_model_types():
    # The salt folds in only the model types bucket_limits actually caps.
    assert max_bucket_resolution_for(ModelType.ANIMA) == ANIMA_MAX_BUCKET_RESOLUTION
    assert max_bucket_resolution_for(ModelType.STABLE_DIFFUSION_XL_10_BASE) is None


def test_image_salt_folds_in_bucket_cap_for_capped_models():
    # An Anima cache built before the cap existed lacked the field, so the capped
    # salt must differ from an otherwise-identical uncapped one. Isolate the cap:
    # `_SameStr` stringifies to "ANIMA" exactly like the real enum (so the
    # model_type field matches) but is not a genuine ModelType member, so the cap
    # helper returns None for it. The only payload difference is the cap field --
    # proving it is actually included in the digest, not silently dropped.
    class _SameStr:
        def __str__(self):
            return str(ModelType.ANIMA)

    capped = _cfg(model_type=ModelType.ANIMA)
    uncapped = _cfg(model_type=_SameStr())
    assert image_cache_salt(capped) == image_cache_salt(_cfg(model_type=ModelType.ANIMA))
    assert image_cache_salt(capped) != image_cache_salt(uncapped)


def test_text_salt_tracks_encoders_and_embeddings():
    assert text_cache_salt(_cfg()) != text_cache_salt(_cfg(text_encoder=_Part("other-clip")))
    assert text_cache_salt(_cfg()) != text_cache_salt(_cfg(text_encoder_layer_skip=2))
    assert text_cache_salt(_cfg()) != text_cache_salt(_cfg(text_encoder_2_sequence_length=256))
    assert text_cache_salt(_cfg()) != text_cache_salt(
        _cfg(additional_embeddings=[_Emb("tok.safetensors", "myted")])
    )
    assert text_cache_salt(_cfg()) != text_cache_salt(_cfg(text_encoder_3=_Part("t5")))


def test_text_salt_ignores_excluded_encoders_and_image_changes():
    # an encoder with include=False must not affect the salt
    assert text_cache_salt(_cfg()) == text_cache_salt(_cfg(text_encoder_3=_Part("garbage", include=False)))
    assert text_cache_salt(_cfg()) == text_cache_salt(_cfg(resolution="768"))
    assert text_cache_salt(_cfg()) == text_cache_salt(_cfg(vae=_Part("/v.safetensors")))


# --------------------------------------------------------------------------- #
# dataset dimension (modules/util/dataset_key.py, folded into both salts)
# --------------------------------------------------------------------------- #

class _Concept:
    def __init__(self, path, enabled=True, include_subdirectories=False):
        self.path = path
        self.enabled = enabled
        self.include_subdirectories = include_subdirectories


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(content)


def _salts(concept_dir):
    """Both salts for a config whose sole concept is ``concept_dir``."""
    dataset_key.reset_memo()  # these tests edit a dataset in place within one process
    cfg = _cfg(concepts=[_Concept(concept_dir)])
    return image_cache_salt(cfg), text_cache_salt(cfg)


def test_salts_unchanged_when_config_carries_no_concepts():
    # The pre-existing _cfg() has no `concepts` attribute at all. Its salts must stay
    # byte-identical to what they were before dataset fingerprinting existed, so
    # deploying this cannot invalidate a cache it has nothing to say about.
    assert image_cache_salt(_cfg()) == image_cache_salt(_cfg(concepts=None))
    assert text_cache_salt(_cfg()) == text_cache_salt(_cfg(concepts=[]))


def test_both_salts_move_when_an_image_is_added():
    """The reported failure: a concept gains a file at an unchanged path. Before
    this, both caches were reused against a dataset that had grown under them."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = os.path.join(tmp_dir, "concept")
        _write(os.path.join(concept_dir, "a.png"), b"image a")
        _write(os.path.join(concept_dir, "a.txt"), "caption a")
        before_image, before_text = _salts(concept_dir)

        _write(os.path.join(concept_dir, "b.png"), b"image b")
        after_image, after_text = _salts(concept_dir)

        assert before_image != after_image, "the latent cache must not be reused"
        assert before_text != after_text, "the text cache must not be reused either"


def test_caption_edit_moves_only_the_text_salt():
    """The split that makes this affordable: rewording a caption must not push every
    image back through the VAE."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_dir = os.path.join(tmp_dir, "concept")
        _write(os.path.join(concept_dir, "a.png"), b"image a")
        _write(os.path.join(concept_dir, "a.txt"), "a cat")
        before_image, before_text = _salts(concept_dir)

        _write(os.path.join(concept_dir, "a.txt"), "a dog")  # same length on purpose
        after_image, after_text = _salts(concept_dir)

        assert before_text != after_text, "the text cache must be re-encoded"
        assert before_image == after_image, "the latent cache must be reused untouched"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
