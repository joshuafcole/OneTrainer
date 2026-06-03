"""Behavioural tests for the latent/text cache salts (modules/util/cache_key.py).

The helpers are pure (stdlib only, duck-typed config), so these run without the
training stack: ``python tests/test_cache_key.py`` or under pytest. They guard the
property that makes reusing a non-cleared cache safe — a salt must move when (and
only when) an input that changes the cached tensors changes.
"""

import importlib.util
import os
import types

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
