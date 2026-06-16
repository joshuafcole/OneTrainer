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
from modules.util.config.SliderConfig import SliderAxisConfig, SliderPromptConfig  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.slider_caption_util import parse_slider_coordinates  # noqa: E402


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
    setup.train_device = torch.device("cpu")
    return setup


class _FakeConfigObj:
    def __init__(self, in_channels):
        self.in_channels = in_channels


class _FakeTransformer:
    def __init__(self, in_channels=16):
        self.config = _FakeConfigObj(in_channels)


class _FakeModel:
    """Records every prompt passed to encode_text and returns a 1-vector tensor
    so the setup's prompt construction is fully observable."""

    def __init__(self, in_channels=16):
        self.transformer = _FakeTransformer(in_channels)
        self.encoded = []

    def encode_text(self, train_device=None, text=None):
        self.encoded.append(text)
        return torch.zeros(1, 1, 4)


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
# coordinate-labeled image regime (docs §10)
# --------------------------------------------------------------------------

def test_parse_extracts_declared_axis_and_strips():
    clean, coords = parse_slider_coordinates("a car on a road, (distance:-2)", ["distance"])
    assert coords == {"distance": -2.0}
    assert clean == "a car on a road", f"axis token not cleanly stripped: {clean!r}"


def test_parse_passes_through_undeclared_emphasis():
    # a1111 emphasis on a non-axis token must survive untouched
    clean, coords = parse_slider_coordinates("(red car:1.2), (distance:3)", ["distance"])
    assert coords == {"distance": 3.0}
    assert "(red car:1.2)" in clean


def test_parse_case_insensitive_float_and_multi_axis():
    clean, coords = parse_slider_coordinates("x, (Distance:+0.5), (Lighting:-1)", ["distance", "lighting"])
    assert coords == {"distance": 0.5, "lighting": -1.0}
    assert clean == "x"


def test_parse_no_declared_axes_is_identity():
    text = "a car, (distance:1)"
    clean, coords = parse_slider_coordinates(text, [])
    assert coords == {} and clean == text


def _axis(name, gain_k=1.0, is_target=True, enabled=True):
    a = SliderAxisConfig.default_values()
    a.name, a.gain_k, a.is_target, a.enabled = name, gain_k, is_target, enabled
    return a


def test_resolve_target_axis_picks_the_single_target():
    config = _config(slider_axes=[
        _axis("distance", gain_k=2.0, is_target=True),
        _axis("lighting", is_target=False),
    ])
    ax = AnimaSliderSetup._resolve_target_axis(config)
    assert ax.name == "distance" and ax.gain_k == 2.0


def test_resolve_target_axis_requires_exactly_one():
    for axes in ([], [_axis("a", is_target=True), _axis("b", is_target=True)]):
        try:
            AnimaSliderSetup._resolve_target_axis(_config(slider_axes=axes))
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError for axes={axes!r}")


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


def test_coordinate_loss_sets_per_sample_multipliers():
    setup = _bare_setup()
    set_calls = []
    seen = []
    targets = [torch.zeros(1, 16, 1, 4, 4), torch.ones(1, 16, 1, 4, 4)]
    multipliers = [-0.5, 1.5]

    def run(i, m):
        seen.append((i, m))
        return torch.zeros(1, 16, 1, 4, 4)  # predict zeros for both samples

    loss = setup._slider_coordinate_loss(run, set_calls.append, targets, multipliers)
    # each sample's multiplier is applied in order, then the adapter is reset to 1.0
    assert set_calls == [-0.5, 1.5, 1.0], f"multiplier schedule wrong: {set_calls}"
    assert seen == [(0, -0.5), (1, 1.5)]
    # mean of mse(0, 0)=0 and mse(0, 1)=1
    assert abs(loss.item() - 0.5) < 1e-6


if __name__ == "__main__":
    test_loader_drives_step_count_and_concept_type()
    test_loader_validation_override()
    test_choose_triple_respects_weights()
    test_build_pairs_bare_pair()
    test_build_pairs_preservation_augments_each_context()
    test_latent_shape_parsing()
    test_caching_encodes_each_prompt_once()
    test_parse_extracts_declared_axis_and_strips()
    test_parse_passes_through_undeclared_emphasis()
    test_parse_case_insensitive_float_and_multi_axis()
    test_parse_no_declared_axes_is_identity()
    test_resolve_target_axis_picks_the_single_target()
    test_resolve_target_axis_requires_exactly_one()
    test_make_flow_target_is_rectified_flow()
    test_coordinate_loss_sets_per_sample_multipliers()
    print("all anima_slider tests passed")
