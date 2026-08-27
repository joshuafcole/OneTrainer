"""Host-contract tests for the two Concept-Sliders setups.

Three things are pinned here, none of which need model weights:

  * **Wiring.** Which (model type, SLIDER) entries exist in the factory, and that
    a model type advertising SLIDER can actually save, load and feed itself.
  * **The datasetless loader contract.** Step count, concept_type, validation.
  * **The hosts' step.** predict() is driven end to end against fake models that
    record every call, so the sequence of multipliers, the number of forwards and
    the exact upstream API each host calls are all asserted.

The fake models mirror the signatures the real AnimaModel / StableDiffusionXLModel
expose today -- a keyword renamed upstream fails here rather than at run time on a
GPU. CPU only. Run with ``python -m pytest tests/test_slider_hosts.py``.
"""

import copy
import os
import sys
from contextlib import nullcontext
from random import Random

import torch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modules.util.create as create  # noqa: E402  (populates the factory registry)
from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader  # noqa: E402
from modules.dataLoader.AnimaSliderImageDataLoader import AnimaSliderImageDataLoader  # noqa: E402
from modules.dataLoader.BaseDataLoader import BaseDataLoader  # noqa: E402
from modules.dataLoader.SliderDataLoader import SliderDataLoader  # noqa: E402
from modules.dataLoader.SliderPromptPairDataLoader import SliderPromptPairDataLoader  # noqa: E402
from modules.modelLoader.BaseModelLoader import BaseModelLoader  # noqa: E402
from modules.modelSaver.BaseModelSaver import BaseModelSaver  # noqa: E402
from modules.modelSetup.AnimaSliderSetup import AnimaSliderSetup  # noqa: E402
from modules.modelSetup.BaseModelSetup import BaseModelSetup  # noqa: E402
from modules.modelSetup.mixin.ModelSetupSliderMixin import ModelSetupSliderMixin  # noqa: E402
from modules.modelSetup.StableDiffusionXLSliderSetup import (  # noqa: E402
    SliderConditioning,
    StableDiffusionXLSliderSetup,
)
from modules.ui.BaseSliderTabView import BaseSliderTabView  # noqa: E402
from modules.ui.CtkSliderTabView import CtkSliderTabView  # noqa: E402
from modules.ui.PySide6SliderTabView import PySide6SliderTabView  # noqa: E402
from modules.ui.SliderAxesWindowController import (  # noqa: E402
    SliderAxesWindowController,
    SliderAxisListController,
)
from modules.ui.SliderTabController import SliderTabController  # noqa: E402
from modules.ui.TopBarController import TopBarController  # noqa: E402
from modules.util import factory  # noqa: E402
from modules.util.config.SliderConfig import SliderAxisConfig, SliderPromptConfig  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ConceptType import ConceptType  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.enum.ModelType import ModelType, PeftType  # noqa: E402
from modules.util.enum.SliderRegime import SliderRegime  # noqa: E402
from modules.util.enum.TrainingMethod import TrainingMethod  # noqa: E402
from modules.util.ModelNames import EmbeddingName, ModelNames  # noqa: E402

SLIDER_MODEL_TYPES = (
    ModelType.ANIMA,
    ModelType.STABLE_DIFFUSION_XL_10_BASE,
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _triple(target="person", positive="old", negative="young", weight=1.0, enabled=True):
    t = SliderPromptConfig.default_values()
    t.target, t.positive, t.negative, t.weight, t.enabled = target, positive, negative, weight, enabled
    return t


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_only_models_with_a_slider_host_advertise_slider():
    """The dropdown gate. ModelType.supported_training_methods() answers in grouped
    tuples -- Anima shares one with six other models, SDXL with eight -- so this is
    the check that the gate really is per-model and not per-group."""
    advertised = {
        model_type for model_type in ModelType
        if TrainingMethod.SLIDER in create.supported_training_methods(model_type)
    }
    assert advertised == set(SLIDER_MODEL_TYPES), f"unexpected slider models: {advertised}"

    # the model types that share Anima's and SDXL's supported_training_methods()
    # tuple, and would come along for free if SLIDER were appended there
    tuple_mates = [
        ModelType.QWEN, ModelType.Z_IMAGE, ModelType.FLUX_2, ModelType.ERNIE,
        ModelType.KREA_2, ModelType.IDEOGRAM_4,
        ModelType.STABLE_DIFFUSION_3, ModelType.WUERSTCHEN_2, ModelType.PIXART_ALPHA,
        ModelType.FLUX_DEV_1, ModelType.SANA, ModelType.HUNYUAN_VIDEO,
        ModelType.HI_DREAM_FULL, ModelType.CHROMA_1,
    ]
    for model_type in tuple_mates:
        assert TrainingMethod.SLIDER not in create.supported_training_methods(model_type), model_type
        assert factory.get(BaseModelSetup, model_type, TrainingMethod.SLIDER) is None, model_type


def test_the_gate_reads_the_factory_not_a_list(monkeypatch):
    """Take the Anima host out of the factory and the dropdown entry goes with it.
    This is what makes 'registering a host is what adds the entry' true rather than
    a convention someone has to remember."""
    real_get = create.factory.get

    def without_the_anima_slider(base_cls, *args, **kwargs):
        if (base_cls, args) == (BaseModelSetup, (ModelType.ANIMA, TrainingMethod.SLIDER)):
            return None
        return real_get(base_cls, *args, **kwargs)

    monkeypatch.setattr(create.factory, "get", without_the_anima_slider)
    assert TrainingMethod.SLIDER not in create.supported_training_methods(ModelType.ANIMA)
    # ... and the other hosts are unaffected: the gate is per model type
    assert TrainingMethod.SLIDER in create.supported_training_methods(
        ModelType.STABLE_DIFFUSION_XL_10_BASE)


@pytest.mark.parametrize("model_type", SLIDER_MODEL_TYPES)
def test_a_slider_model_can_save_load_and_feed_itself(model_type):
    """Every piece a slider run needs at the far end of training. The saver and the
    loader do NOT fall back to a model-type-only entry, and supported_output_formats
    raises on an unknown method -- so all three of these fail only after a full run
    if they are missing."""
    assert factory.get(BaseModelSetup, model_type, TrainingMethod.SLIDER) is not None
    assert factory.get(BaseModelSaver, model_type, TrainingMethod.SLIDER) is not None
    assert factory.get(BaseModelLoader, model_type, TrainingMethod.SLIDER) is not None
    assert factory.get(BaseDataLoader, model_type, TrainingMethod.SLIDER) is SliderDataLoader

    formats = model_type.supported_output_formats(TrainingMethod.SLIDER)
    assert formats == model_type.supported_lora_formats()
    assert ModelFormat.DIFFUSERS_LORA in formats


def test_slider_saver_is_the_models_own_lora_saver():
    """Not just 'a saver' -- the same one LoRA training uses, since a slider file
    is a LoRA file."""
    for model_type in SLIDER_MODEL_TYPES:
        for base_cls in (BaseModelSaver, BaseModelLoader):
            assert factory.get(base_cls, model_type, TrainingMethod.SLIDER) \
                is factory.get(base_cls, model_type, TrainingMethod.LORA)


# ---------------------------------------------------------------------------
# datasetless loader
# ---------------------------------------------------------------------------

def _loader(config, is_validation=False):
    return SliderPromptPairDataLoader(
        torch.device("cpu"), torch.device("cpu"), config,
        model=None, model_setup=None, train_progress=None, is_validation=is_validation,
    )


def test_loader_drives_the_step_count_and_keeps_the_prior_paths_inert():
    loader = _loader(_config(slider_steps_per_epoch=7, batch_size=2))

    dataset = loader.get_data_set()
    assert dataset.approximate_length() == 7
    dataset.start_next_epoch()
    assert dataset.epoch == 0

    batches = list(loader.get_data_loader())
    assert len(batches) == 7
    for batch in batches:
        # all-STANDARD: the trainer selects prior-prediction and counterexample rows
        # by concept type, so both extra frozen forwards stay unrun.
        assert batch["concept_type"] == [ConceptType.STANDARD.value] * 2
        assert all(ConceptType(c) == ConceptType.STANDARD for c in batch["concept_type"])
    assert len(loader.get_data_loader()) == 7


def test_the_slot_dispatches_on_the_regime():
    """One factory entry, two regimes with opposite needs: PROMPT_PAIR has no
    dataset at all, IMAGE wants the model's whole MGDS pipeline. The factory keys
    on (model type, training method) and cannot see a regime, so the dispatch has
    to happen at construction -- and it is picked here rather than by an
    `if` inside a loader that would then have to be both things."""
    assert SliderDataLoader._create_impl(
        _config(model_type=ModelType.ANIMA, slider_regime=SliderRegime.PROMPT_PAIR),
    ) is SliderPromptPairDataLoader
    assert SliderDataLoader._create_impl(
        _config(model_type=ModelType.ANIMA, slider_regime=SliderRegime.IMAGE),
    ) is AnimaSliderImageDataLoader


def test_a_model_without_an_image_regime_loader_says_so_instead_of_falling_back():
    """SDXL has a prompt-pair slider host and no coordinate-labeled one. Falling
    back to its ordinary loader would emit no slider_coordinate, and the run would
    die on a missing batch key several minutes in, after the model load."""
    with pytest.raises(NotImplementedError, match="image slider regime is not implemented"):
        SliderDataLoader._create_impl(_config(
            model_type=ModelType.STABLE_DIFFUSION_XL_10_BASE,
            slider_regime=SliderRegime.IMAGE,
        ))


def test_validation_is_empty_rather_than_broken():
    """The trainer's validation pass reads concept_name / concept_path /
    concept_seed off a batch. A prompt-pair slider has no held-out data, so it
    reports a zero-length epoch and the pass returns before touching one."""
    loader = _loader(_config(slider_steps_per_epoch=500, batch_size=4), is_validation=True)
    assert loader.get_data_set().approximate_length() == 0
    assert list(loader.get_data_loader()) == []


# ---------------------------------------------------------------------------
# host-neutral plumbing (shared by both hosts)
# ---------------------------------------------------------------------------

def _bare(setup_cls):
    setup = object.__new__(setup_cls)  # bypass the device-needing __init__
    setup.train_device = torch.device("cpu")
    return setup


@pytest.mark.parametrize("setup_cls", [AnimaSliderSetup, StableDiffusionXLSliderSetup])
def test_prompt_pairs_bare_and_preserved(setup_cls):
    setup = _bare(setup_cls)

    pos, neg = setup._slider_prompt_pairs(_triple(), _config(slider_preservation_prompts=""))
    assert (pos, neg) == (["old"], ["young"]), "no preservation set => the bare pair (CS Eq. 7)"

    pos, neg = setup._slider_prompt_pairs(
        _triple(), _config(slider_preservation_prompts="a man | a woman"))
    # the bare pair is always included, so a context widens the average
    assert pos == ["old", "old, a man", "old, a woman"]
    assert neg == ["young", "young, a man", "young, a woman"]


@pytest.mark.parametrize("setup_cls", [AnimaSliderSetup, StableDiffusionXLSliderSetup])
def test_preservation_set_accepts_both_separators(setup_cls):
    """The same field is a single-line entry on one toolkit and a text box on the
    other; a user who pressed Enter must not get one context containing a newline."""
    setup = _bare(setup_cls)
    for text in ("a man|a woman", "a man\na woman", "a man\n a woman ", "a man | a woman"):
        pos, _ = setup._slider_prompt_pairs(_triple(), _config(slider_preservation_prompts=text))
        assert pos == ["old", "old, a man", "old, a woman"], text


def test_choose_triple_respects_weights():
    setup = _bare(AnimaSliderSetup)
    triples = [_triple(positive="a+", weight=0.0), _triple(positive="b+", weight=1.0)]
    chosen = {setup._choose_triple(triples, Random(i)).positive for i in range(50)}
    assert chosen == {"b+"}, f"a zero-weight triple must never be chosen: {chosen}"

    # all-zero weights plainly mean "no preference", not "an error"
    both_zero = [_triple(positive="a+", weight=0.0), _triple(positive="b+", weight=0.0)]
    chosen = {setup._choose_triple(both_zero, Random(i)).positive for i in range(50)}
    assert chosen == {"a+", "b+"}


def test_triples_must_be_enabled():
    setup = _bare(AnimaSliderSetup)
    kept = setup._slider_triples(_config(slider_prompts=[
        _triple(positive="kept"), _triple(positive="skipped", enabled=False),
    ]))
    assert [t.positive for t in kept] == ["kept"]
    with pytest.raises(RuntimeError, match="at least one enabled prompt pair"):
        setup._slider_triples(_config(slider_prompts=[_triple(enabled=False)]))
    with pytest.raises(RuntimeError):
        setup._slider_triples(_config(slider_prompts=[]))


def test_resolution_parsing():
    setup = _bare(AnimaSliderSetup)
    assert setup._slider_resolution(_config(resolution="512")) == (512, 512)
    assert setup._slider_resolution(_config(resolution="512x768")) == (512, 768)
    assert setup._slider_resolution(_config(resolution="512,1024")) == (512, 512)


def test_noise_level_range_is_order_insensitive_and_clamped():
    setup = _bare(AnimaSliderSetup)
    config = _config(slider_sigma_min=0.9, slider_sigma_max=0.1)  # swapped by the user
    values = [setup._slider_sample_noise_level(config, Random(i)) for i in range(50)]
    assert all(0.1 <= v <= 0.9 for v in values), "swapped bounds must give the range meant"
    assert max(values) - min(values) > 0.3, "the range must actually be sampled"

    config = _config(slider_sigma_min=-1.0, slider_sigma_max=5.0)
    assert all(0.0 <= setup._slider_sample_noise_level(config, Random(i)) <= 1.0 for i in range(20))


def test_conditioning_is_encoded_once_per_prompt():
    setup = _bare(AnimaSliderSetup)
    calls = []
    encode = lambda text: (calls.append(text), torch.zeros(1))[1]  # noqa: E731
    setup._slider_cached_conditioning("hello", encode)
    setup._slider_cached_conditioning("hello", encode)
    setup._slider_cached_conditioning("world", encode)
    assert calls == ["hello", "world"], "a frozen prompt must not be re-encoded every step"


# ---------------------------------------------------------------------------
# fake models: the upstream API surface each host actually calls
# ---------------------------------------------------------------------------

class _FakeWrapper:
    """Stands in for LoRAModuleWrapper. Records every multiplier it is set to."""

    def __init__(self):
        self.multiplier = 1.0
        self.history = []

    def set_multiplier(self, multiplier: float):
        self.multiplier = float(multiplier)
        self.history.append(float(multiplier))


class _FakeDataType:
    @staticmethod
    def torch_dtype():
        return torch.float32


class _Config(dict):
    """Mirrors diffusers' FrozenDict: readable as a dict AND as attributes, which
    is how upstream reads a scheduler config in both styles."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _FakeAnimaTransformer:
    def __init__(self, in_channels=16):
        self.config = _Config(in_channels=in_channels)
        self.calls = []

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, padding_mask, return_dict):
        assert return_dict is False
        self.calls.append({
            "hidden_states": hidden_states, "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states, "padding_mask": padding_mask,
        })
        # a conditioning-dependent, deterministic "velocity"
        scale = encoder_hidden_states.sum()
        return (hidden_states * 0.1 + scale * 0.01,)


class _FakeAnimaModel:
    LATENT_SCALE = 3.0

    def __init__(self, in_channels=16):
        self.transformer = _FakeAnimaTransformer(in_channels)
        self.transformer_lora = _FakeWrapper()
        self.train_dtype = _FakeDataType()
        self.autocast_context = nullcontext()
        self.encoded = []
        self.encode_calls = []

    def encode_text(
            self, train_device=None, batch_size=1, rand=None, text=None,
            tokens=None, tokens_mask=None, t5_tokens=None, t5_tokens_mask=None,
            text_encoder_output=None, text_encoder_dropout_probability=None,
    ):
        # exactly the AnimaModel.encode_text keywords the two hosts use between
        # them -- prompt-pair passes `text`, the image regime passes the batch's
        # cached tokens. A keyword renamed upstream is a TypeError here.
        self.encode_calls.append({
            "batch_size": batch_size, "tokens": tokens, "tokens_mask": tokens_mask,
            "t5_tokens": t5_tokens, "t5_tokens_mask": t5_tokens_mask,
            "text_encoder_output": text_encoder_output,
            "text_encoder_dropout_probability": text_encoder_dropout_probability,
        })
        if text is not None:
            self.encoded.append(text)
            torch.manual_seed(abs(hash(text)) % (2 ** 31))
            return torch.randn(1, 8, 4, requires_grad=True)
        torch.manual_seed(5)
        return torch.randn(batch_size, 8, 4, requires_grad=True)

    def scale_latents(self, latents):
        # a scale the tests can see, so "the host forgot to scale" is a failure
        return latents * self.LATENT_SCALE


class _FakeUnetOutput:
    def __init__(self, sample):
        self.sample = sample


class _FakeUnet:
    def __init__(self, in_channels=4):
        self.config = _Config(in_channels=in_channels)
        self.calls = []

    def __call__(self, *, sample, timestep, encoder_hidden_states, added_cond_kwargs):
        self.calls.append({
            "sample": sample, "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": added_cond_kwargs,
        })
        scale = encoder_hidden_states.sum() + added_cond_kwargs["text_embeds"].sum()
        return _FakeUnetOutput(sample * 0.1 + scale * 0.01)


class _FakeScheduler:
    def __init__(self, prediction_type='epsilon', num_train_timesteps=1000):
        self.config = _Config(prediction_type=prediction_type,
                              num_train_timesteps=num_train_timesteps)
        self.betas = torch.linspace(1e-4, 2e-2, num_train_timesteps)


class _FakeSdxlModel:
    def __init__(self, prediction_type='epsilon'):
        self.unet = _FakeUnet()
        self.unet_lora = _FakeWrapper()
        self.noise_scheduler = _FakeScheduler(prediction_type)
        self.train_dtype = _FakeDataType()
        self.autocast_context = nullcontext()
        self.encoded = []

    def encode_text(self, train_device=None, batch_size=1, text=None):
        # exactly the StableDiffusionXLModel.encode_text keywords this host uses
        self.encoded.append(text)
        torch.manual_seed(abs(hash(text)) % (2 ** 31))
        return torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280)

    def combine_text_encoder_output(self, te1, te2, pooled):
        return torch.concat([te1, te2], dim=-1), pooled


class _Progress:
    global_step = 3


def _slider_config(**overrides):
    values = {
        "slider_prompts": [_triple()],
        "slider_eta": 2.0,
        "slider_strength": 0.75,
        "slider_symmetric": True,
        "slider_anchor_steps": 0,
        "resolution": "64",
    }
    values.update(overrides)
    return _config(**values)


# ---------------------------------------------------------------------------
# the hosts' training step
# ---------------------------------------------------------------------------

def test_anima_predict_runs_the_objective():
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    config = _slider_config()

    data = setup.predict(model, {}, config, _Progress())
    loss = setup.calculate_loss(model, {}, data, config)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # one base forward + one bare c+/c- pair at multiplier 0, then both poles
    assert model.transformer_lora.history == [0.0, 0.0, 0.75, -0.75, 1.0]
    assert len(model.transformer.calls) == 5

    call = model.transformer.calls[0]
    assert call["hidden_states"].shape == (1, 16, 1, 8, 8), "5D (B,C,T,H,W) latent at 64px"
    assert call["padding_mask"].shape == (1, 1, 64, 64), "the padding mask is pixel-space"
    assert call["timestep"].shape == (1,)


def test_anima_encodes_every_prompt_once_and_caches_across_steps():
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    config = _slider_config(slider_preservation_prompts="a man")

    setup.predict(model, {}, config, _Progress())
    # target + bare pair + one preservation context each side
    assert model.encoded == ["person", "old", "old, a man", "young", "young, a man"]

    setup.predict(model, {}, config, _Progress())
    assert model.encoded == ["person", "old", "old, a man", "young", "young, a man"], \
        "the second step must reuse the cache, not re-run the text encoder"
    # 1 base + 2 pairs, all frozen, then the two trained poles
    assert len(model.transformer.calls) == 2 * (1 + 2 * 2 + 2)


def test_anima_anchor_steps_add_frozen_forwards():
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    setup.predict(model, {}, _slider_config(slider_anchor_steps=4), _Progress())
    # the SDEdit walk zeroes the adapter itself, then the objective zeroes it again
    assert model.transformer_lora.history == [0.0, 0.0, 0.75, -0.75, 1.0]
    # 4 anchor forwards, then base + one c+/c- pair + both trained poles
    assert len(model.transformer.calls) == 4 + 5


def test_sdxl_predict_runs_the_objective():
    setup = _bare(StableDiffusionXLSliderSetup)
    model = _FakeSdxlModel()
    config = _slider_config()

    data = setup.predict(model, {}, config, _Progress())
    loss = setup.calculate_loss(model, {}, data, config)
    assert loss.ndim == 0 and torch.isfinite(loss)

    assert model.unet_lora.history == [0.0, 0.0, 0.75, -0.75, 1.0]
    assert len(model.unet.calls) == 5

    call = model.unet.calls[0]
    assert call["sample"].shape == (1, 4, 8, 8), "4D latent at 64px"
    assert call["encoder_hidden_states"].shape == (1, 77, 768 + 1280), "both encoders concatenated"
    time_ids = call["added_cond_kwargs"]["time_ids"]
    assert time_ids.tolist() == [[64, 64, 0, 0, 64, 64]], "original size, no crop, target size"
    assert call["added_cond_kwargs"]["text_embeds"].shape == (1, 1280)


def test_sdxl_conditioning_carries_both_halves():
    setup = _bare(StableDiffusionXLSliderSetup)
    model = _FakeSdxlModel()
    a = setup._encode(model, "one")
    b = setup._encode(model, "two")
    assert isinstance(a, SliderConditioning)
    assert not torch.equal(a.pooled_embeds, b.pooled_embeds), \
        "the pooled embedding varies with the prompt and must travel with it"
    assert not a.prompt_embeds.requires_grad, "the frozen conditioning must be detached"


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_sdxl_anchor_walks_the_trajectory_for_both_parameterizations(prediction_type):
    """A zero-terminal-SNR SDXL is forced to v_prediction, so the anchor's DDIM step
    has to convert. Both must produce a finite x_t and the same call count."""
    setup = _bare(StableDiffusionXLSliderSetup)
    model = _FakeSdxlModel(prediction_type)
    setup.predict(model, {}, _slider_config(slider_anchor_steps=3), _Progress())
    assert model.unet_lora.history == [0.0, 0.0, 0.75, -0.75, 1.0]
    assert len(model.unet.calls) == 3 + 5
    assert torch.isfinite(model.unet.calls[-1]["sample"]).all()


def test_sdxl_refuses_embedding_training():
    """An embedding cannot train through a conditioning that is encoded once and
    detached. Switching it on takes several deliberate steps, so refuse rather than
    silently do something else -- an embedding whose gradient never arrives is a
    run that reports a falling loss and produces nothing."""
    config = _slider_config()
    embedding = type(config.embedding).default_values()
    embedding.train = True
    embedding.is_output_embedding = False
    config.additional_embeddings = [embedding]
    assert config.train_any_embedding(), "the fixture must actually request embedding training"
    with pytest.raises(RuntimeError, match="cannot train embeddings"):
        StableDiffusionXLSliderSetup._check_trainable_parts(config)


def test_sdxl_text_encoder_training_warns_rather_than_refusing(capsys):
    """text_encoder.train defaults to ON, so a user arriving from LoRA training has
    asked for nothing. Refusing on a default is hostile; silence would leave a
    visible switch doing nothing. Warn, and build only the UNet adapter."""
    assert TrainConfig.default_values().text_encoder.train is True, \
        "the premise of this test: text encoder training is on by default"

    config = _slider_config()
    assert config.text_encoder.train
    StableDiffusionXLSliderSetup._check_trainable_parts(config)
    assert "does not train the text encoders" in capsys.readouterr().out

    config.text_encoder.train = False
    config.text_encoder_2.train = False
    StableDiffusionXLSliderSetup._check_trainable_parts(config)
    assert capsys.readouterr().out == "", "no notice when nothing was asked for"


@pytest.mark.parametrize("setup_cls,model_factory,wrapper_attr", [
    (AnimaSliderSetup, _FakeAnimaModel, "transformer_lora"),
    (StableDiffusionXLSliderSetup, _FakeSdxlModel, "unet_lora"),
])
def test_symmetric_off_drops_the_negative_pole(setup_cls, model_factory, wrapper_attr):
    setup = _bare(setup_cls)
    model = model_factory()
    setup.predict(model, {}, _slider_config(slider_symmetric=False), _Progress())
    assert getattr(model, wrapper_attr).history == [0.0, 0.0, 0.75, 1.0]


@pytest.mark.parametrize("setup_cls,model_factory", [
    (AnimaSliderSetup, _FakeAnimaModel),
    (StableDiffusionXLSliderSetup, _FakeSdxlModel),
])
def test_predict_refuses_an_empty_prompt_list(setup_cls, model_factory):
    setup = _bare(setup_cls)
    with pytest.raises(RuntimeError, match="at least one enabled prompt pair"):
        setup.predict(model_factory(), {}, _slider_config(slider_prompts=[]), _Progress())


def test_lora_weight_dtype_default_is_untouched():
    """A guard on the config surface, not the slider: the slider fields must not
    have displaced anything in TrainConfig's default ordering."""
    assert TrainConfig.default_values().lora_weight_dtype == DataType.FLOAT_32


# ---------------------------------------------------------------------------
# the PEFT-type gate
#
# DoRA, OFT and weight-decomposed LoKr recompose the base weight instead of
# adding a scaled delta, so LoRAModule raises for any multiplier but 1.0 -- and
# it raises in the *forward*. Without a setup-time check that is a config
# mistake you pay a full model load to discover, the same shape of trap as a
# training method missing from supported_output_formats().
# ---------------------------------------------------------------------------

_SLIDER_INCOMPATIBLE_PEFT = [
    ({"peft_type": PeftType.OFT_2}, "orthogonal rotation"),
    ({"peft_type": PeftType.LORA, "lora_decompose": True}, "DoRA"),
    ({"peft_type": PeftType.LOKR, "lokr_weight_decompose": True}, "weight-decomposed LoKr"),
]

_SLIDER_COMPATIBLE_PEFT = [
    {"peft_type": PeftType.LORA, "lora_decompose": False},
    {"peft_type": PeftType.LOHA},
    {"peft_type": PeftType.LOKR, "lokr_weight_decompose": False},
]


@pytest.mark.parametrize("overrides,reason", _SLIDER_INCOMPATIBLE_PEFT)
def test_peft_types_without_a_signed_multiplier_are_refused(overrides, reason):
    with pytest.raises(RuntimeError, match="signed multiplier") as excinfo:
        ModelSetupSliderMixin._check_slider_peft_type(_slider_config(**overrides))
    assert reason in str(excinfo.value), "the message must name what is wrong, not just that it is"


@pytest.mark.parametrize("overrides", _SLIDER_COMPATIBLE_PEFT)
def test_additive_peft_types_are_accepted(overrides):
    ModelSetupSliderMixin._check_slider_peft_type(_slider_config(**overrides))


@pytest.mark.parametrize("overrides,_reason", _SLIDER_INCOMPATIBLE_PEFT)
@pytest.mark.parametrize("setup_cls", [AnimaSliderSetup, StableDiffusionXLSliderSetup])
def test_both_hosts_check_the_peft_type_before_building_an_adapter(setup_cls, overrides, _reason):
    """setup_model is the first per-run hook a setup gets. Passing None for the
    model is the assertion: the check has to fire before anything touches it."""
    with pytest.raises(RuntimeError, match="signed multiplier"):
        _bare(setup_cls).setup_model(None, _slider_config(**overrides))


def test_the_refused_set_is_exactly_the_set_that_raises_in_the_forward():
    """The gate and LoRAModule must not drift apart. A PEFT type that stops (or
    starts) supporting a signed multiplier has to change both, and letting one
    move alone is either a false refusal or the late failure this gate exists to
    prevent."""
    refused = set()
    for peft_type in PeftType:
        for decompose in (False, True):
            config = _slider_config(
                peft_type=peft_type, lora_decompose=decompose, lokr_weight_decompose=decompose)
            try:
                ModelSetupSliderMixin._check_slider_peft_type(config)
            except RuntimeError:
                refused.add((peft_type, decompose))

    assert refused == {
        (PeftType.OFT_2, False), (PeftType.OFT_2, True),
        (PeftType.LORA, True),
        (PeftType.LOKR, True),
    }


def _svd_config(svd_dtype, **overrides):
    config = _slider_config(**overrides)
    config.quantization.svd_dtype = svd_dtype
    return config


def test_svdquant_is_refused_even_on_an_additive_peft_type():
    """The fourth way into the same forward raise, and the one the PeftType loop
    above cannot see.

    quantization.svd_dtype turns the *base* linears into BaseLinearSVD, whose
    forward_with_lora refuses any multiplier but 1.0 -- so plain LoRA, which the
    gate otherwise accepts, still dies mid-run. Not a PEFT type, so it needs its
    own branch and its own test; the set-comparison test iterates PeftType and
    would never have found it.
    """
    for overrides in _SLIDER_COMPATIBLE_PEFT:
        with pytest.raises(RuntimeError, match="signed multiplier") as excinfo:
            ModelSetupSliderMixin._check_slider_peft_type(
                _svd_config(DataType.FLOAT_8, **overrides))
        assert "SVD dtype" in str(excinfo.value), "the message must name the control that is set"


def test_svd_dtype_none_is_the_only_accepted_value():
    """Every non-NONE SVD dtype builds SVD linears, so NONE is the whole allowed
    set. Enumerated rather than spot-checked: a new DataType member added later
    is refused by default, which is the safe direction for a guard whose failure
    mode is a crash several minutes into a run."""
    accepted = {d for d in DataType
                if not _raises(lambda d=d: ModelSetupSliderMixin._check_slider_peft_type(
                    _svd_config(d, peft_type=PeftType.LORA, lora_decompose=False)))}
    assert accepted == {DataType.NONE}


def _raises(fn):
    try:
        fn()
    except RuntimeError:
        return True
    return False


@pytest.mark.parametrize("setup_cls", [AnimaSliderSetup, StableDiffusionXLSliderSetup])
def test_both_hosts_refuse_svdquant_before_touching_the_model(setup_cls):
    with pytest.raises(RuntimeError, match="signed multiplier"):
        _bare(setup_cls).setup_model(None, _svd_config(DataType.FLOAT_8))


# ---------------------------------------------------------------------------
# resuming from a backup
#
# A slider is a LoRA on disk, so it backs up as one and must resume as one.
# Falling through to the fine-tune branch hands an adapter directory to the
# base-model loader: a resume that cannot work but reads like one that should.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("training_method", [TrainingMethod.LORA, TrainingMethod.SLIDER])
def test_backup_resume_targets_the_lora_name_for_adapter_methods(training_method):
    model_names = ModelNames(base_model="the-base-model")
    model_names.set_backup_path(training_method, "/workspace/backup/2026-01-01")

    assert model_names.lora == "/workspace/backup/2026-01-01"
    assert model_names.base_model == "the-base-model", "the base model must survive the resume"


def test_backup_resume_still_routes_fine_tunes_and_embeddings():
    fine_tune = ModelNames(base_model="the-base-model")
    fine_tune.set_backup_path(TrainingMethod.FINE_TUNE, "/backup/ft")
    assert fine_tune.base_model == "/backup/ft"
    assert fine_tune.lora == ""

    embedding = ModelNames(embedding=EmbeddingName("uuid", "original"))
    embedding.set_backup_path(TrainingMethod.EMBEDDING, "/backup/emb")
    assert embedding.embedding.model_name == "/backup/emb"
    assert embedding.base_model == ""


def test_every_training_method_has_a_backup_destination():
    """No silent fall-through: a method added later must be considered here."""
    for training_method in TrainingMethod:
        model_names = ModelNames(embedding=EmbeddingName("uuid", ""))
        model_names.set_backup_path(training_method, "/backup/x")
        assert "/backup/x" in (
            model_names.lora, model_names.base_model, model_names.embedding.model_name)


# ---------------------------------------------------------------------------
# the UI surfaces
# ---------------------------------------------------------------------------

def test_every_training_method_has_a_dropdown_label():
    """TopBarController indexes a dict by TrainingMethod. A method without an
    entry is a KeyError that takes down the whole top bar, and only for the model
    types that advertise it -- so it can ship looking fine."""
    config = _config(model_type=ModelType.ANIMA, training_method=TrainingMethod.SLIDER)
    controller = TopBarController.__new__(TopBarController)
    controller.train_config = config

    for model_type in ModelType:
        labels = dict(controller.get_training_methods(model_type))
        assert labels, f"{model_type} offers no training method at all"

    anima = [method for _, method in controller.get_training_methods(ModelType.ANIMA)]
    assert TrainingMethod.SLIDER in anima
    sd15 = [method for _, method in controller.get_training_methods(ModelType.STABLE_DIFFUSION_15)]
    assert TrainingMethod.SLIDER not in sd15


def test_every_regime_declares_its_blocks_and_its_hint():
    """The regime selector hides every block not named for the current regime. A
    regime missing from the map therefore renders an *empty* tab rather than
    failing -- so the map is checked against the enum, not the other way round."""
    for regime in SliderRegime:
        assert BaseSliderTabView.blocks_for_regime(regime), f"{regime} shows no settings block"
        assert BaseSliderTabView.hint_for_regime(regime), f"{regime} has no hint text"


def test_the_prompt_pair_regime_shows_the_prompt_list():
    blocks = BaseSliderTabView.blocks_for_regime(SliderRegime.PROMPT_PAIR)
    assert "prompt_list" in blocks, "the triples are the whole input of this regime"
    assert "prompt_pair" in blocks


def test_the_image_regime_hides_the_prompt_list():
    """Its input is the Concepts tab. Leaving the triples on screen would say they
    were being read, and they are not."""
    blocks = BaseSliderTabView.blocks_for_regime(SliderRegime.IMAGE)
    assert "image" in blocks
    assert "prompt_list" not in blocks
    assert "prompt_pair" not in blocks


def test_every_regime_can_be_selected():
    """The other half of the block map: a regime with blocks but no dropdown entry
    is unreachable, and one with an entry but no blocks renders an empty tab."""
    controller = SliderTabController(_config())
    offered = [regime for _label, regime in controller.get_slider_regimes()]
    assert offered == list(SliderRegime)


def test_both_toolkits_can_open_the_axes_editor():
    """The tab hides which toolkit it is on behind _open_axes_window, so a missing
    implementation is an abstract-method TypeError at tab construction -- which
    only happens for a user who picked SLIDER."""
    for view_cls in (CtkSliderTabView, PySide6SliderTabView):
        assert not view_cls.__abstractmethods__, view_cls.__abstractmethods__
        assert "_open_axes_window" in vars(view_cls)


def test_the_axes_editor_edits_the_configs_own_axis_list():
    """Not a copy: the window is opened from the tab and dismissed, and whatever
    the user typed has to be on the TrainConfig that gets saved."""
    config = _config()
    controller = SliderAxesWindowController(config)
    assert controller.config is config

    axes = SliderAxisListController(config)
    axis = axes.create_new_element()
    assert axis.name == "", "a pre-filled axis name is a row that looks configured and trains nothing"

    # a cloned row must not keep the original's uuid, or the two rows are one row
    clone = axes.randomize_uuid(copy.deepcopy(axis))
    assert clone.uuid != axis.uuid


# ---------------------------------------------------------------------------
# the coordinate-labeled image loader
#
# The pipeline is driven for real -- _load_input_modules builds the actual MGDS
# nodes and their map_fns are run -- so what is asserted is what the loader built,
# not what a test handed it. The MGDS boundary itself (an image on disk, a VAE) is
# the only thing stubbed.
# ---------------------------------------------------------------------------

def _axis(name, gain_k=1.0, is_target=True, enabled=True):
    axis = SliderAxisConfig.default_values()
    axis.name, axis.gain_k, axis.is_target, axis.enabled = name, gain_k, is_target, enabled
    return axis


def _image_loader(*axes):
    loader = object.__new__(AnimaSliderImageDataLoader)
    config = _config(
        model_type=ModelType.ANIMA,
        slider_regime=SliderRegime.IMAGE,
        slider_axes=list(axes),
    )
    return loader, config


def _coordinate_nodes(loader, config):
    """The two nodes the loader appends, found by what they produce rather than by
    position -- upstream may add input modules of its own."""
    modules = loader._load_input_modules(config, DataType.FLOAT_32, vae_frame_dim=True)
    extract = [m for m in modules if getattr(m, "out_name", None) == "slider_coordinate"]
    strip = [m for m in modules
             if getattr(m, "out_name", None) == "prompt" and getattr(m, "in_name", None) == "prompt"]
    assert len(extract) == 1 and len(strip) == 1
    return modules, extract[0], strip[0]


def test_the_loader_reads_the_coordinate_and_strips_it_from_the_caption():
    loader, config = _image_loader(_axis("distance", gain_k=4.0))
    _modules, extract, strip = _coordinate_nodes(loader, config)

    caption = "a photo of a car on a road, (distance:-2)"
    assert torch.equal(extract.map_fn(caption), torch.tensor([-2.0]))
    assert strip.map_fn(caption) == "a photo of a car on a road"


def test_the_coordinate_is_read_before_the_caption_is_stripped():
    """Order inside the input stage, which no per-node test can see. Strip first
    and the extractor is handed a caption the coordinate has already left, so every
    sample reads 0.0 and the run trains nothing while reporting a falling loss.

    Both nodes are in the input stage for the other half of the ordering: tag
    dropout runs later, and must not get a chance to delete a coordinate token
    before it is removed deliberately."""
    loader, config = _image_loader(_axis("distance"))
    modules, extract, strip = _coordinate_nodes(loader, config)
    assert modules.index(extract) < modules.index(strip)


def test_the_gain_is_not_baked_into_the_cached_coordinate():
    """gain_k is applied at step time, so retuning it does not invalidate a latent
    cache that can take an hour to rebuild."""
    loader, config = _image_loader(_axis("distance", gain_k=4.0))
    _modules, extract, _strip = _coordinate_nodes(loader, config)
    assert torch.equal(extract.map_fn("x, (distance:-2)"), torch.tensor([-2.0]))


def test_a_caption_with_no_coordinate_is_zero_rather_than_an_error():
    """One unlabelled image must not take the run down mid-epoch. It trains
    nothing (the objective drops it) and a whole batch of them is refused there,
    where the message can name the axis."""
    loader, config = _image_loader(_axis("distance"))
    _modules, extract, strip = _coordinate_nodes(loader, config)
    assert torch.equal(extract.map_fn("a photo of a car"), torch.tensor([0.0]))
    assert strip.map_fn("a photo of a car") == "a photo of a car"


def test_a_declared_confounder_is_stripped_but_does_not_supply_the_coordinate():
    """The reason to declare a second axis at all: keep it out of the conditioning
    on a run that is not training it."""
    loader, config = _image_loader(_axis("distance"), _axis("age", is_target=False))
    _modules, extract, strip = _coordinate_nodes(loader, config)

    caption = "a portrait, (distance:-2), (age:60)"
    assert torch.equal(extract.map_fn(caption), torch.tensor([-2.0]))
    assert strip.map_fn(caption) == "a portrait"


def test_an_unusable_axis_set_fails_while_the_dataset_is_being_wired():
    """Not per image inside MGDS, and not after the model load: _load_input_modules
    resolves the axes once, so a config mistake surfaces with the config still on
    screen."""
    loader, config = _image_loader(_axis("distance"), _axis("age"))
    with pytest.raises(RuntimeError, match="Exactly one slider axis"):
        loader._load_input_modules(config, DataType.FLOAT_32, vae_frame_dim=True)


def test_the_coordinate_survives_the_caching_boundary():
    """A per-sample tensor missing from the cache split is silently absent from the
    batch rather than an error -- the run gets several minutes in and then dies on
    a KeyError. So the split, the sort and the output list are all checked."""
    loader, config = _image_loader(_axis("distance"))
    captured = {}

    loader._cache_modules_from_names = lambda *a, **kw: captured.update(kw) or []
    loader._output_modules_from_out_names = lambda *a, **kw: captured.update(kw) or []
    model = _FakeAnimaModel()
    model.vae = None
    loader._cache_modules(config, model=model, model_setup=None)
    loader._output_modules(config, model=model, model_setup=None)

    assert "slider_coordinate" in captured["image_split_names"]
    assert "slider_coordinate" in captured["sort_names"]
    assert "slider_coordinate" in captured["output_names"]


def test_the_ordinary_anima_loader_adds_nothing():
    """The hook the loader above uses lives on AnimaBaseDataLoader, so this is the
    check that it changed nothing for every other Anima training method."""
    loader = object.__new__(AnimaBaseDataLoader)
    assert loader._additional_split_names(_config(model_type=ModelType.ANIMA)) == []


# ---------------------------------------------------------------------------
# the IMAGE-regime step
#
# predict() is driven for real against a fake AnimaModel that records every call,
# so what is asserted is the sequence of multipliers the host set, the tensors it
# built, and the exact upstream API it called -- not values a test handed it.
# ---------------------------------------------------------------------------

def _image_config(coords_axis="distance", gain_k=1.0, **overrides):
    values = {
        "model_type": ModelType.ANIMA,
        "slider_regime": SliderRegime.IMAGE,
        "slider_axes": [_axis(coords_axis, gain_k=gain_k)],
        "slider_sigma_min": 0.2,
        "slider_sigma_max": 0.8,
    }
    values.update(overrides)
    config = _config(**values)
    # Anima builds no text-encoder adapter, so a slider run leaves this off and
    # the text cache is live. (With it on, train_text_encoder_or_embedding()
    # disables the cache and Qwen3 runs every step for nothing -- which is
    # upstream's behaviour for an Anima LoRA too, and not this slice's to change.)
    config.text_encoder.train = False
    return config


def _image_batch(coords, in_channels=16, latent_hw=8, seq=4):
    n = len(coords)
    torch.manual_seed(3)
    return {
        "latent_image": torch.randn(n, in_channels, 1, latent_hw, latent_hw),
        "slider_coordinate": torch.tensor(coords, dtype=torch.float32).reshape(n, 1),
        "tokens": torch.zeros(n, seq, dtype=torch.long),
        "tokens_mask": torch.ones(n, seq, dtype=torch.long),
        "t5_tokens": torch.zeros(n, seq, dtype=torch.long),
        "t5_tokens_mask": torch.ones(n, seq, dtype=torch.long),
        "text_encoder_hidden_state": torch.randn(n, 8, 4),
    }


def test_anima_image_predict_runs_the_coordinate_objective():
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    batch = _image_batch([-2.0, 1.0])

    data = setup.predict(model, batch, _image_config(), _Progress())
    loss = setup.calculate_loss(model, batch, data, _image_config())
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad

    # one forward per distinct multiplier, then back to the resting multiplier.
    # No frozen-base pass and no 0.0 anywhere: this objective has no base target.
    assert model.transformer_lora.history == [-2.0, 1.0, 1.0]
    assert len(model.transformer.calls) == 2

    call = model.transformer.calls[0]
    assert call["hidden_states"].shape == (1, 16, 1, 8, 8), "5D (B,C,T,H,W) latent"
    assert call["padding_mask"].shape == (1, 1, 64, 64), "the padding mask is pixel-space"
    assert call["timestep"].shape == (1,)


def test_anima_image_step_groups_the_poles_into_one_forward_each():
    """The binary-pole case, which is what an explicit image-pair regime would
    have been: a batch of eight is two forwards, not eight."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    batch = _image_batch([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    setup.predict(model, batch, _image_config(), _Progress())

    assert model.transformer_lora.history == [1.0, -1.0, 1.0]
    assert len(model.transformer.calls) == 2
    assert model.transformer.calls[0]["hidden_states"].shape[0] == 4
    assert model.transformer.calls[1]["hidden_states"].shape[0] == 4


def test_anima_image_applies_the_gain_at_step_time():
    """gain_k lives outside the cache, so this is where it has to be read. The
    coordinates are as authored; only the multiplier is scaled."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    batch = _image_batch([-2.0, 2.0])

    setup.predict(model, batch, _image_config(gain_k=0.25), _Progress())
    assert model.transformer_lora.history == [-0.5, 0.5, 1.0]


def test_anima_image_step_feeds_the_cached_conditioning_and_does_not_drop_it():
    """Two things at once. The batch's cached conditioner output is passed
    through (no live Qwen3 forward per step), and caption dropout is off -- this
    regime rests on the caption explaining everything the axis does not, so
    dropping it would leave the whole image as the residual for the adapter to
    absorb."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    batch = _image_batch([1.0, -1.0])

    setup.predict(model, batch, _image_config(), _Progress())

    assert len(model.encode_calls) == 1, "one encode for the whole batch"
    call = model.encode_calls[0]
    assert call["batch_size"] == 2
    assert call["text_encoder_output"] is batch["text_encoder_hidden_state"]
    assert call["tokens"] is batch["tokens"]
    assert call["t5_tokens"] is batch["t5_tokens"]
    assert call["text_encoder_dropout_probability"] is None


def test_anima_image_x_t_is_the_real_image_latent_scaled():
    """At sigma 0 the flow forward is the identity, so x_t must be exactly the
    scaled latent. That pins two things a plausible-looking step gets wrong:
    training on noise instead of the image, and forgetting that the dataloader
    caches the UNSCALED VAE mean."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    batch = _image_batch([1.0])
    config = _image_config(slider_sigma_min=0.0, slider_sigma_max=0.0)

    setup.predict(model, batch, config, _Progress())

    x_t = model.transformer.calls[0]["hidden_states"]
    assert torch.allclose(x_t, batch["latent_image"] * _FakeAnimaModel.LATENT_SCALE, atol=1e-6)
    assert torch.allclose(model.transformer.calls[0]["timestep"], torch.zeros(1))


def test_anima_image_draws_a_noise_level_per_image_not_per_batch():
    """Rows are grouped by multiplier, so one sigma for the whole batch would tie
    the noise level to the coordinate and let the adapter learn the wrong one."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    setup.predict(model, _image_batch([1.0] * 6), _image_config(), _Progress())

    timesteps = model.transformer.calls[0]["timestep"]
    assert timesteps.shape == (6,)
    assert len(set(timesteps.tolist())) == 6, f"all rows share a sigma: {timesteps}"
    assert all(0.2 <= t <= 0.8 for t in timesteps.tolist())


def test_anima_image_batch_with_no_coordinates_is_refused_by_name():
    """End to end: the mistyped-axis case reaches the user as an error about the
    axis name, not as a silently untrained run."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    with pytest.raises(RuntimeError, match="axis name"):
        setup.predict(model, _image_batch([0.0, 0.0]), _image_config(), _Progress())


def test_the_image_regime_is_reached_only_from_its_own_regime():
    """The prompt-pair step must not be handed an image batch, or vice versa: the
    two read entirely different keys off it."""
    setup = _bare(AnimaSliderSetup)
    model = _FakeAnimaModel()
    config = _slider_config(model_type=ModelType.ANIMA)
    setup.predict(model, {}, config, _Progress())
    assert model.transformer_lora.history[0] == 0.0, "prompt-pair still runs the frozen base"


def test_setup_model_refuses_an_unusable_axis_set_before_building_the_adapter():
    """Same bar as the PEFT-type check beside it: a config mistake should not cost
    a model load and a dataset cache to discover."""
    setup = _bare(AnimaSliderSetup)
    config = _image_config()
    config.slider_axes = [_axis("distance"), _axis("age")]

    # model=None: reaching AnimaLoRASetup.setup_model at all would be an
    # AttributeError, so a RuntimeError proves the refusal came first.
    with pytest.raises(RuntimeError, match="Exactly one slider axis"):
        setup.setup_model(model=None, config=config)


class _Part:
    def __init__(self):
        self.mode = None

    def eval(self):
        self.mode = "eval"

    def train(self):
        self.mode = "train"


class _ResidencyModel:
    def __init__(self):
        self.transformer = _Part()
        self.text_encoder = _Part()
        self.text_conditioner = _Part()
        self.vae = _Part()
        self.materialized = None

    def materialize_only(self, *parts):
        self.materialized = parts


@pytest.mark.parametrize("regime,expected", [
    (SliderRegime.PROMPT_PAIR, ("transformer", "text_encoder", "vae")),
    (SliderRegime.IMAGE, ("transformer",)),
])
def test_the_image_regime_keeps_the_ordinary_lora_residency(regime, expected):
    """Not cosmetic: a prompt-pair slider has to keep Qwen3 resident because it
    encodes its prompts live, and pinning that on a coordinate slider -- which
    caches its text like any dataset run -- would hold a text encoder and a VAE on
    the train device for a whole run that never calls them."""
    setup = _bare(AnimaSliderSetup)
    model = _ResidencyModel()
    config = _image_config(slider_regime=regime, latent_caching=True)
    config.transformer.train = True
    config.text_encoder.train = False

    setup.setup_train_device(model, config)
    assert model.materialized == expected
