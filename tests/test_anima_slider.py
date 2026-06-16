"""Tests for the Anima slider wiring that don't need a real model forward.

Two parts:
  * the datasetless prompt-pair loader contract (step count, concept_type shape,
    validation override) -- pure bookkeeping, no torch model;
  * AnimaSliderSetup's pure helpers (weighted triple selection, preservation-pair
    construction incl. the bare pair, latent-shape parsing) with encode_text
    stubbed so prompt construction is observable.

The heavy quantization import chain is stubbed like the other suites:
``python tests/test_anima_slider.py`` or pytest.
"""

import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_stub = types.ModuleType("modules.util.quantization_util")
_stub.get_unquantized_weight = lambda m, dtype, device: m.weight.detach().to(dtype)
_stub.get_weight_shape = lambda m: m.weight.shape
_stub.quantize_layers = lambda *a, **k: None
sys.modules["modules.util.quantization_util"] = _stub

from modules.dataLoader.AnimaSliderDataLoader import AnimaSliderDataLoader  # noqa: E402
from modules.modelSetup.AnimaSliderSetup import AnimaSliderSetup  # noqa: E402
from modules.util.config.SliderConfig import SliderImagePairConfig, SliderPromptConfig  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.enum.SliderRegime import SliderRegime  # noqa: E402


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


# --------------------------------------------------------------------------
# loader contract
# --------------------------------------------------------------------------

def test_loader_drives_step_count_and_concept_type():
    config = _config(slider_steps_per_epoch=7, batch_size=2)
    loader = AnimaSliderDataLoader(
        torch.device("cpu"), torch.device("cpu"), config,
        model=None, model_setup=None, train_progress=None,
    )

    ds = loader.get_data_set()
    assert ds.approximate_length() == 7
    ds.start_next_epoch()  # must not raise; advances the epoch counter
    assert ds.epoch == 0

    batches = list(loader.get_data_loader())
    assert len(batches) == 7, f"expected 7 step batches, got {len(batches)}"
    for batch in batches:
        assert batch["concept_type"] == [ConceptType.STANDARD.value] * 2
        # all-STANDARD keeps the trainer's prior-prediction path inert
        assert all(ConceptType(c) == ConceptType.STANDARD for c in batch["concept_type"])
    # defensive read in the train loop: no latent_image present
    assert batches[0].get("latent_image") is None


def test_loader_validation_override():
    config = _config(slider_steps_per_epoch=500, batch_size=4)
    loader = AnimaSliderDataLoader(
        torch.device("cpu"), torch.device("cpu"), config,
        model=None, model_setup=None, train_progress=None, is_validation=True,
    )
    assert loader.get_data_set().approximate_length() == 1
    batches = list(loader.get_data_loader())
    assert len(batches) == 1
    assert batches[0]["concept_type"] == [ConceptType.STANDARD.value]


# --------------------------------------------------------------------------
# setup helpers (no model forward)
# --------------------------------------------------------------------------

def _bare_setup() -> AnimaSliderSetup:
    setup = object.__new__(AnimaSliderSetup)  # bypass the device-needing __init__
    setup._cond_cache = {}
    setup._latent_cache = {}
    setup.train_device = torch.device("cpu")
    return setup


class _FakeConfigObj:
    def __init__(self, in_channels):
        self.in_channels = in_channels


class _FakeTransformer:
    def __init__(self, in_channels=16):
        self.config = _FakeConfigObj(in_channels)


class _FakeLatentDist:
    def __init__(self, mean):
        self.mean = mean


class _FakeEncoded:
    def __init__(self, mean):
        self.latent_dist = _FakeLatentDist(mean)


class _FakeVAE:
    """Records encode() input shapes and returns a zero latent downscaled by 8,
    so _encode_image's preprocessing + caching is observable without a real VAE."""

    def __init__(self, z_dim=16):
        self.z_dim = z_dim
        self.encoded_shapes = []

    def encode(self, px):
        self.encoded_shapes.append(tuple(px.shape))
        b, _c, t, h, w = px.shape
        return _FakeEncoded(torch.zeros(b, self.z_dim, t, h // 8, w // 8))


class _FakeDtype:
    def torch_dtype(self):
        return torch.float32


class _FakeModel:
    """Records every prompt passed to encode_text and returns a 1-vector tensor
    so the setup's prompt construction is fully observable."""

    def __init__(self, in_channels=16):
        self.transformer = _FakeTransformer(in_channels)
        self.vae = _FakeVAE()
        self.train_dtype = _FakeDtype()
        self.encoded = []

    def encode_text(self, train_device=None, text=None):
        self.encoded.append(text)
        return torch.zeros(1, 1, 4)

    def scale_latents(self, latents):
        # identity stand-in for the mean/std normalization (irrelevant here)
        return latents


def _triple(target, positive, negative, weight=1.0, enabled=True):
    t = SliderPromptConfig.default_values()
    t.target, t.positive, t.negative, t.weight, t.enabled = target, positive, negative, weight, enabled
    return t


def test_choose_triple_respects_weights():
    import random
    setup = _bare_setup()
    triples = [_triple("c", "a+", "a-", weight=0.0), _triple("c", "b+", "b-", weight=1.0)]
    # zero-weight triple must never be chosen when a positive-weight one exists
    chosen = {setup._choose_triple(triples, random.Random(i)).positive for i in range(50)}
    assert chosen == {"b+"}, f"weighting ignored: {chosen}"


def test_build_pairs_bare_pair():
    setup = _bare_setup()
    model = _FakeModel()
    config = _config(slider_preservation_prompts="")
    pos, neg = setup._build_pairs(model, _triple("person", "old", "young"), config)
    assert len(pos) == 1 and len(neg) == 1
    assert model.encoded == ["old", "young"]


def test_build_pairs_preservation_augments_each_context():
    setup = _bare_setup()
    model = _FakeModel()
    config = _config(slider_preservation_prompts="a man | a woman")
    pos, neg = setup._build_pairs(model, _triple("person", "old", "young"), config)
    # bare pair + one pair per preservation context, equal length both sides
    assert len(pos) == len(neg) == 3
    assert "old" in model.encoded and "old, a man" in model.encoded and "old, a woman" in model.encoded
    assert "young, a woman" in model.encoded


def test_latent_shape_parsing():
    setup = _bare_setup()
    model = _FakeModel(in_channels=16)
    assert setup._latent_shape(model, _config(resolution="512")) == (16, 64, 64)
    assert setup._latent_shape(model, _config(resolution="512x768")) == (16, 64, 96)
    # multi-resolution list: first entry wins
    assert setup._latent_shape(model, _config(resolution="512,1024")) == (16, 64, 64)


def test_caching_encodes_each_prompt_once():
    setup = _bare_setup()
    model = _FakeModel()
    setup._encode_cached(model, "hello")
    setup._encode_cached(model, "hello")
    setup._encode_cached(model, "world")
    assert model.encoded == ["hello", "world"], "encode_text should run once per unique prompt"


# --------------------------------------------------------------------------
# image-pair regime
# --------------------------------------------------------------------------

def _image_pair(before, after, prompt="", weight=1.0, enabled=True):
    p = SliderImagePairConfig.default_values()
    p.before, p.after, p.prompt, p.weight, p.enabled = before, after, prompt, weight, enabled
    return p


def test_choose_pair_respects_weights():
    import random
    setup = _bare_setup()
    pairs = [_image_pair("a0", "b0", weight=0.0), _image_pair("a1", "b1", weight=1.0)]
    chosen = {setup._choose_pair(pairs, random.Random(i)).before for i in range(50)}
    assert chosen == {"a1"}, f"weighting ignored: {chosen}"


def test_make_flow_target_is_rectified_flow():
    setup = _bare_setup()
    x0 = torch.randn(1, 16, 1, 8, 8)
    sigma = 0.3
    x_t, target = setup._make_flow_target(x0, sigma, torch.float32, torch.Generator().manual_seed(0))
    # target = noise - x0  =>  noise = target + x0; x_t must be (1-σ)x0 + σ·noise
    noise = target + x0
    expected_xt = (1.0 - sigma) * x0 + sigma * noise
    assert torch.allclose(x_t, expected_xt, atol=1e-5), "x_t is not the rectified-flow interpolation"
    assert x_t.shape == x0.shape and target.shape == x0.shape


def test_encode_image_preprocesses_and_caches():
    from PIL import Image as PILImage
    setup = _bare_setup()
    model = _FakeModel(in_channels=16)
    config = _config(resolution="512")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.png")
        PILImage.new("RGB", (64, 96), (128, 40, 200)).save(path)  # odd aspect -> cover-crop

        latent_a = setup._encode_image(model, path, config)
        latent_b = setup._encode_image(model, path, config)  # cache hit

    # encoded exactly once, at the configured pixel resolution as a 5D (B,C,T,H,W) tensor
    assert len(model.vae.encoded_shapes) == 1, "VAE should encode each path once"
    assert model.vae.encoded_shapes[0] == (1, 3, 1, 512, 512)
    # latent downscaled by 8, with the transformer channel count
    assert tuple(latent_a.shape) == (1, 16, 1, 64, 64)
    assert latent_a is latent_b  # cached object returned verbatim


def test_image_pair_predict_requires_a_pair():
    setup = _bare_setup()
    config = _config(slider_regime=SliderRegime.IMAGE_PAIR, slider_image_pairs=[])
    try:
        setup._predict_image_pair(model=None, config=config, train_progress=None, deterministic=True)
    except RuntimeError as e:
        assert "image" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError when no enabled image pairs are configured")


if __name__ == "__main__":
    test_loader_drives_step_count_and_concept_type()
    test_loader_validation_override()
    test_choose_triple_respects_weights()
    test_build_pairs_bare_pair()
    test_build_pairs_preservation_augments_each_context()
    test_latent_shape_parsing()
    test_caching_encodes_each_prompt_once()
    test_choose_pair_respects_weights()
    test_make_flow_target_is_rectified_flow()
    test_encode_image_preprocesses_and_caches()
    test_image_pair_predict_requires_a_pair()
    print("all anima_slider tests passed")
