"""Anima textual-inversion training: the token has to actually train, and it has to survive the save.

Anima trains a TI token into the Qwen3 word-embedding table and nowhere else. Two paths use it -- a
placeholder riding alongside a transformer LoRA (TrainingMethod.LORA) and pure textual inversion
(TrainingMethod.EMBEDDING) -- and both have the same three ways to be silently wrong:

  1. The token is never handed to the optimizer, or never wired into the encoder, so it stays at its
     seed while the run reports healthy losses.
  2. Qwen3's cached conditioner output is left switched on, so the step-time graph never touches the
     word-embedding table and no gradient can reach the token.
  3. The training and the inference-time tokenization disagree. Anima's conditioner takes T5 ids as
     queries, and ComfyUI's Anima encoder injects NO TI vector on the T5 side -- so anything trained
     into the T5 table cannot be reproduced there. This port leaves T5 entirely alone; these tests pin
     that, because "left alone" is invisible in a loss curve.

Plus the one that only shows up at save time: bundled TI vectors survive the COMFY/KOHYA converters
only because AnimaModel declares lora_text_encoders(). Without it, `convert(..., strict=True)` raises
"No conversion found for key bundle_emb...". Everything works until someone saves for ComfyUI.

Every model here is a real (tiny) Qwen3 / AnimaTextConditioner / CosmosTransformer3DModel /
AutoencoderKLQwenImage on CPU, and the tokenizers are real HF fast tokenizers built in memory -- no
stand-ins for the parts under test.

Run with ``PYTHONPATH=. python -m pytest tests/test_anima_embedding_training.py``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from diffusers import (  # noqa: E402
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from safetensors.torch import load_file  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model  # noqa: E402

from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader  # noqa: E402
from modules.model.AnimaModel import AnimaModel  # noqa: E402
from modules.modelLoader.AnimaModelLoader import AnimaEmbeddingModelLoader  # noqa: E402
from modules.modelLoader.BaseModelLoader import BaseModelLoader  # noqa: E402
from modules.modelSaver.anima.AnimaEmbeddingSaver import AnimaEmbeddingSaver  # noqa: E402
from modules.modelSaver.anima.AnimaLoRASaver import AnimaLoRASaver  # noqa: E402
from modules.modelSaver.AnimaEmbeddingModelSaver import AnimaEmbeddingModelSaver  # noqa: E402
from modules.modelSaver.BaseModelSaver import BaseModelSaver  # noqa: E402
from modules.modelSetup.AnimaEmbeddingSetup import AnimaEmbeddingSetup  # noqa: E402
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup  # noqa: E402
from modules.modelSetup.BaseModelSetup import BaseModelSetup  # noqa: E402
from modules.util import factory  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.enum.ModelType import ModelType  # noqa: E402
from modules.util.enum.TrainingMethod import TrainingMethod  # noqa: E402
from modules.util.TrainProgress import TrainProgress  # noqa: E402

CPU = torch.device("cpu")
PLACEHOLDER = "<smoketoken>"
TOKEN_COUNT = 2
WORDS = ["a", "photo", "of", "red", "square", "on", "black", "background", PLACEHOLDER]


def _tokenizer() -> PreTrainedTokenizerFast:
    # A real HF fast tokenizer, built in memory: add_tokens / __len__ / padding all behave as they do
    # for Qwen2Tokenizer, which is what AdditionalEmbeddingWrapper and _add_embeddings_to_tokenizer
    # depend on. A hand-written stand-in would be a second implementation of the thing under test.
    vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", *WORDS])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


def _tiny_model(config: TrainConfig) -> AnimaModel:
    model = AnimaModel(ModelType.ANIMA)
    model.train_config = config
    model.tokenizer = _tokenizer()
    model.t5_tokenizer = _tokenizer()
    model.text_encoder = Qwen3Model(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=1024,
    ))
    model.text_conditioner = AnimaTextConditioner(
        source_dim=16, target_dim=16, model_dim=16, num_layers=1,
        num_attention_heads=2, target_vocab_size=64, min_sequence_length=8,
    )
    model.transformer = CosmosTransformer3DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=16,
        num_layers=1, text_embed_dim=16, adaln_lora_dim=8, max_size=(4, 8, 8),
        patch_size=(1, 2, 2), encoder_hidden_states_channels=16,
    )
    model.vae = AutoencoderKLQwenImage(
        base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[False],
        latents_mean=[0.0] * 4, latents_std=[1.0] * 4,
    )
    # left at construction defaults: .timesteps then has one entry per training timestep, which is the
    # index space _get_timestep_discrete draws from.
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    return model


def _config(training_method: TrainingMethod) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.ANIMA
    config.training_method = training_method
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.train_dtype = DataType.FLOAT_32
    config.fallback_train_dtype = DataType.FLOAT_32
    config.embedding_weight_dtype = DataType.FLOAT_32
    config.latent_caching = True
    config.text_encoder.train_embedding = True
    config.text_encoder.train = False
    config.transformer.train = True
    config.preserve_embedding_norm = False
    config.embedding_learning_rate = 1e-2
    config.embedding.placeholder = PLACEHOLDER
    config.embedding.train = True
    config.embedding.token_count = TOKEN_COUNT
    config.embedding.initial_embedding_text = "square"
    config.embedding.is_output_embedding = False
    return config


def _lora_config() -> TrainConfig:
    # The joint path: the placeholder is an *additional* embedding riding alongside the LoRA, which is
    # the configuration AnimaLoRASetup used to ignore outright.
    config = _config(TrainingMethod.LORA)
    additional = type(config.embedding).default_values()
    additional.placeholder = PLACEHOLDER
    additional.train = True
    additional.token_count = TOKEN_COUNT
    additional.initial_embedding_text = "square"
    additional.is_output_embedding = False
    config.additional_embeddings = [additional]
    return config


def _setup(config: TrainConfig):
    cls = AnimaEmbeddingSetup if config.training_method == TrainingMethod.EMBEDDING else AnimaLoRASetup
    return cls(train_device=CPU, temp_device=CPU, debug_mode=False)


def _prepared(config: TrainConfig) -> tuple[AnimaModel, object]:
    model = _tiny_model(config)
    setup = _setup(config)
    setup.setup_model(model, config)
    return model, setup


def _trained_embedding(model: AnimaModel):
    # the one embedding these fixtures configure, whichever path put it there
    return model.all_text_encoder_embeddings()[0]


def _preparation_modules(config: TrainConfig, model: AnimaModel) -> list:
    # _preparation_modules is a pure function of (config, model); BaseDataLoader.__init__ builds the whole
    # mgds pipeline, which needs a dataset on disk.
    return AnimaBaseDataLoader.__new__(AnimaBaseDataLoader)._preparation_modules(config, model)


def _batch(model: AnimaModel, prompt: str = f"a photo of {PLACEHOLDER}") -> dict:
    # what AnimaBaseDataLoader hands predict(): the placeholder substituted on the Qwen branch only,
    # both token sets present, and no cached conditioner output.
    qwen = model.tokenizer(
        [model.add_text_encoder_embeddings_to_prompt(prompt)],
        max_length=16, padding="max_length", truncation=True, return_tensors="pt",
    )
    t5 = model.t5_tokenizer(
        [prompt], max_length=16, padding="max_length", truncation=True, return_tensors="pt",
    )
    return {
        "latent_image": torch.randn(1, 4, 1, 8, 8),
        "loss_weight": torch.ones(1),
        "tokens": qwen.input_ids,
        "tokens_mask": qwen.attention_mask,
        "t5_tokens": t5.input_ids,
        "t5_tokens_mask": t5.attention_mask,
    }


# --------------------------------------------------------------------------------------------------
# 1. the token reaches the optimizer and receives gradient -- the exact bug the LoRA path had
# --------------------------------------------------------------------------------------------------

def test_lora_path_hands_the_placeholder_to_the_optimizer():
    config = _lora_config()
    model, _ = _prepared(config)

    vector = _trained_embedding(model).vector
    optimizer_params = [p for group in model.optimizer.param_groups for p in group["params"]]

    assert any(p is vector for p in optimizer_params), \
        "the trained TI vector is not one of the optimizer's parameters -- it can never move"
    assert vector.requires_grad
    assert any(name.startswith("embeddings/") for name in model.parameters.unique_name_mapping)


def test_lora_path_delivers_gradient_to_the_placeholder():
    config = _lora_config()
    model, setup = _prepared(config)
    vector = _trained_embedding(model).vector

    data = setup.predict(model, _batch(model), config, TrainProgress(), deterministic=True)
    setup.calculate_loss(model, _batch(model), data, config).backward()

    assert vector.grad is not None, "no gradient reached the TI vector"
    assert float(vector.grad.abs().sum()) > 0.0, "the TI vector's gradient is all zeros"


def test_embedding_path_delivers_gradient_to_the_placeholder():
    config = _config(TrainingMethod.EMBEDDING)
    model, setup = _prepared(config)
    vector = _trained_embedding(model).vector

    data = setup.predict(model, _batch(model), config, TrainProgress(), deterministic=True)
    setup.calculate_loss(model, _batch(model), data, config).backward()

    assert vector.grad is not None and float(vector.grad.abs().sum()) > 0.0


def test_a_cached_conditioner_output_is_ignored_while_a_token_is_trained():
    # The gate. If predict() were to consume batch['text_encoder_hidden_state'], Qwen3 would never run
    # at step time and the word-embedding table would be outside the graph -- gradient zero, loss fine.
    config = _config(TrainingMethod.EMBEDDING)
    assert config.train_text_encoder_or_embedding()
    model, setup = _prepared(config)
    vector = _trained_embedding(model).vector

    batch = _batch(model)
    batch["text_encoder_hidden_state"] = torch.randn(1, 512, 16)

    data = setup.predict(model, batch, config, TrainProgress(), deterministic=True)
    setup.calculate_loss(model, batch, data, config).backward()

    assert vector.grad is not None and float(vector.grad.abs().sum()) > 0.0, \
        "the cached conditioner output was consumed, so the token is outside the training graph"


def test_the_cached_conditioner_output_is_still_used_when_no_token_is_trained():
    # The other side of the gate. Turning Qwen3 on unconditionally would be just as wrong: a plain LoRA
    # run keeps the cached conditioner output and must never pay for a live encode.
    config = _config(TrainingMethod.LORA)
    assert not config.train_text_encoder_or_embedding()
    model, setup = _prepared(config)

    called = []
    handle = model.text_encoder.register_forward_pre_hook(lambda *_: called.append(True))
    try:
        batch = _batch(model)
        batch["text_encoder_hidden_state"] = torch.randn(1, 512, 16)
        setup.predict(model, batch, config, TrainProgress(), deterministic=True)
    finally:
        handle.remove()

    assert not called, "Qwen3 ran even though a cached conditioner output was available"


def test_the_dataloader_drops_the_text_cache_while_a_token_is_trained():
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)

    modules = _preparation_modules(config, model)
    assert not any(type(m).__name__ == "EncodeAnimaText" for m in modules), \
        "the conditioner-output encode is still in the pipeline, so its output would be cached"


# --------------------------------------------------------------------------------------------------
# 2. the ComfyUI save round trip -- the lora_text_encoders() guard
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model_format", [ModelFormat.COMFY_LORA, ModelFormat.KOHYA_LORA])
def test_a_bundled_token_survives_the_save_round_trip(tmp_path, model_format):
    config = _lora_config()
    config.bundle_additional_embeddings = True
    model, _ = _prepared(config)

    vector = _trained_embedding(model).vector
    destination = str(tmp_path / f"anima_{model_format}.safetensors")
    AnimaLoRASaver().save(model, model_format, destination, torch.float32)

    written = load_file(destination)
    key = f"bundle_emb.{PLACEHOLDER}.qwen"
    assert key in written, \
        f"the bundled TI vector is missing from the {model_format} file: {sorted(written)[:5]}"
    assert torch.allclose(written[key], vector.detach().to(torch.float32))
    assert any(k.startswith("diffusion_model.") or k.startswith("lora_unet") for k in written), \
        "the adapter itself did not make it into the file"


def test_the_model_declares_the_namespace_the_bundle_key_lives_in():
    # The mechanism behind the round trip above: lora_original_conversion / lora_kohya_conversion append
    # the bundle_emb passthrough only for a model that declares a text encoder, and both converts run
    # strict. Declared without a COMFY_LORA name on purpose -- see AnimaModel.lora_text_encoders.
    model = _tiny_model(_config(TrainingMethod.LORA))
    declared = model.lora_text_encoders()

    assert declared, "AnimaModel declares no text encoder, so bundle_emb.* matches no conversion rule"
    module, names = declared[0]
    assert module is model.text_encoder
    assert names[ModelFormat.DIFFUSERS_LORA] == "text_encoder"
    assert ModelFormat.KOHYA_LORA in names
    assert ModelFormat.COMFY_LORA not in names


# --------------------------------------------------------------------------------------------------
# 3. T5 is left untouched -- what makes the checkpoint reproducible in ComfyUI
# --------------------------------------------------------------------------------------------------

def test_no_placeholder_tokens_are_added_to_the_t5_tokenizer():
    config = _config(TrainingMethod.EMBEDDING)
    model = _tiny_model(config)
    before = len(model.t5_tokenizer)

    _setup(config).setup_model(model, config)

    assert len(model.tokenizer) == before + TOKEN_COUNT, "the Qwen tokenizer did not get the token"
    assert len(model.t5_tokenizer) == before, "placeholder tokens were added to the T5 tokenizer"


def test_no_wrapper_is_hooked_onto_the_conditioners_embedding_table():
    config = _config(TrainingMethod.EMBEDDING)
    model = _tiny_model(config)
    original_forward = model.text_conditioner.embed.forward

    _setup(config).setup_model(model, config)

    assert model.text_conditioner.embed.forward == original_forward, \
        "something replaced AnimaTextConditioner.embed.forward -- the T5 table is being trained"
    assert not hasattr(model, "t5_embedding_wrapper")


def test_no_t5_parameter_group_and_no_t5_knob():
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)

    assert not any("t5" in name for name in model.parameters.unique_name_mapping), \
        "a T5-side parameter group exists"
    # There is deliberately no train_t5_embedding flag: its `true` branch produces a checkpoint whose
    # concept cannot be reproduced in ComfyUI, which is not a choice worth offering.
    assert not hasattr(config, "train_t5_embedding")


def test_encode_text_gives_the_conditioner_the_unsubstituted_t5_ids():
    # The runtime half of the same rule, caught where it actually matters: whatever ids reach
    # AnimaTextConditioner as target_input_ids are the ones ComfyUI's encoder will produce for the same
    # prompt. Substituting the placeholder on this side would train against ids no inference tool emits.
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)
    prompt = f"a photo of {PLACEHOLDER}"

    qwen_text = model.add_text_encoder_embeddings_to_prompt(prompt)
    assert PLACEHOLDER not in qwen_text, "the Qwen branch did not substitute the placeholder"

    seen = {}
    original_forward = model.text_conditioner.forward

    def capture(*args, **kwargs):
        seen["target_input_ids"] = kwargs["target_input_ids"]
        seen["source_input_len"] = kwargs["source_hidden_states"].shape[1]
        return original_forward(*args, **kwargs)

    model.text_conditioner.forward = capture
    try:
        model.encode_text(train_device=CPU, text=[prompt], batch_size=1)
    finally:
        model.text_conditioner.forward = original_forward

    expected = model.t5_tokenizer(
        [prompt], max_length=seen["source_input_len"], padding="max_length", truncation=True,
        return_tensors="pt",
    ).input_ids
    assert torch.equal(seen["target_input_ids"], expected), \
        "the T5 ids are not the ones the raw prompt tokenizes to"


def test_the_dataloader_substitutes_on_the_qwen_branch_only():
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)

    modules = _preparation_modules(config, model)
    by_type = {}
    for module in modules:
        by_type.setdefault(type(module).__name__, []).append(module)

    map_data = by_type["MapData"]
    assert len(map_data) == 1
    assert map_data[0].in_name == "prompt"
    assert map_data[0].out_name == "prompt_qwen"
    assert map_data[0].map_fn == model.add_text_encoder_embeddings_to_prompt

    tokenize = {m.tokens_out_name: m for m in by_type["Tokenize"]}
    assert tokenize["tokens"].in_name == "prompt_qwen", "Qwen3 is not tokenizing the substituted prompt"
    assert tokenize["t5_tokens"].in_name == "prompt", "T5 is tokenizing the substituted prompt"


# --------------------------------------------------------------------------------------------------
# 4. pure TI freezes everything else
# --------------------------------------------------------------------------------------------------

def test_pure_ti_freezes_transformer_conditioner_qwen3_and_vae():
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)

    for name in ("transformer", "text_conditioner", "text_encoder", "vae"):
        component = getattr(model, name)
        trainable = [n for n, p in component.named_parameters() if p.requires_grad]
        assert not trainable, f"{name} has trainable parameters in a pure-TI run: {trainable[:3]}"

    assert model.transformer_lora is None, "a LoRA adapter was built for a pure-TI run"
    assert _trained_embedding(model).vector.requires_grad, "the token itself is frozen"

    optimizer_params = [p for group in model.optimizer.param_groups for p in group["params"]]
    assert len(optimizer_params) == 1, "a pure-TI run must optimize the token and nothing else"


# --------------------------------------------------------------------------------------------------
# 5. the refusals
# --------------------------------------------------------------------------------------------------

def test_output_embeddings_are_refused_rather_than_silently_dead():
    config = _config(TrainingMethod.EMBEDDING)
    config.embedding.is_output_embedding = True
    model = _tiny_model(config)

    with pytest.raises(NotImplementedError, match="Output embeddings"):
        _setup(config).setup_model(model, config)


def test_encode_text_without_t5_tokens_says_so():
    config = _config(TrainingMethod.LORA)
    model = _tiny_model(config)
    tokens = model.tokenizer(["a photo of red square"], return_tensors="pt")

    with pytest.raises(ValueError, match="T5 token ids"):
        model.encode_text(
            train_device=CPU,
            tokens=tokens.input_ids,
            tokens_mask=tokens.attention_mask,
        )


# --------------------------------------------------------------------------------------------------
# 6. the method exists, and a saved token comes back
# --------------------------------------------------------------------------------------------------

def test_anima_advertises_embedding_training_and_resolves_the_whole_trio():
    # The entry point. Everything below can be perfect and the feature still be unreachable: the method
    # only appears in the top bar if ModelType advertises it, and the trainer only runs if the factory
    # resolves a loader, a setup and a saver for (ANIMA, EMBEDDING).
    assert TrainingMethod.EMBEDDING in ModelType.ANIMA.supported_training_methods()

    key = (ModelType.ANIMA, TrainingMethod.EMBEDDING)
    assert factory.get(BaseModelSetup, *key) is AnimaEmbeddingSetup
    assert factory.get(BaseModelLoader, *key) is AnimaEmbeddingModelLoader
    assert factory.get(BaseModelSaver, *key) is AnimaEmbeddingModelSaver


def test_a_saved_token_reloads_into_the_same_vector(tmp_path):
    # The half of the round trip the LoRA bundle test cannot see: a standalone embedding file is only
    # useful if the key it is written under is the key the resume path reads back. Both sides say "qwen"
    # -- the Qwen3 word table being the only table Anima trains into -- and nothing else pins that pair,
    # so a rename on either side would produce a file that saves and resumes to a fresh seed instead.
    config = _config(TrainingMethod.EMBEDDING)
    model, _ = _prepared(config)
    # the loader leaves a None entry for an embedding that has no file yet; the saver reads that dict
    model.embedding_state_dicts = {config.embedding.uuid: None}

    trained = _trained_embedding(model).vector
    with torch.no_grad():
        trained.add_(1.0)  # move it well away from anything _create_new_embedding would seed

    destination = str(tmp_path / "token.safetensors")
    AnimaEmbeddingSaver().save_single(model, ModelFormat.SAFETENSORS, destination, torch.float32)

    written = load_file(destination)
    assert set(written) == {"qwen"}, \
        f"the standalone file does not use the key the bundle and the resume path use: {sorted(written)}"

    resumed = _tiny_model(config)
    resumed.embedding_state_dicts = {config.embedding.uuid: written}
    _setup(config).setup_model(resumed, config)

    assert torch.allclose(_trained_embedding(resumed).vector, trained.detach()), \
        "resuming from the saved file did not restore the trained vector"


# --------------------------------------------------------------------------------------------------
# 7. the two settings that are invisible until a real run goes wrong
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make_config", [lambda: _config(TrainingMethod.EMBEDDING), _lora_config],
                         ids=["embedding", "lora"])
def test_the_embed_table_is_promoted_on_both_paths(make_config):
    # fp16 rows and an fp32 trained vector cannot be concatenated in the wrapper, and a token left in
    # fp16 does not move: the per-step update lands below the representable step for a vocab vector.
    # Same promotion, both training methods. The joint LoRA path is the easier one to forget, and its
    # symptom is the quiet one: the run trains, the loss moves (the adapter is learning), and the token
    # sits still because an fp16 row cannot represent the update.
    config = make_config()
    model = _tiny_model(config)
    model.text_encoder.get_input_embeddings().to(dtype=torch.float16)

    _setup(config).setup_model(model, config)

    assert model.text_encoder.get_input_embeddings().weight.dtype == torch.float32
    assert _trained_embedding(model).vector.dtype == torch.float32


@pytest.mark.parametrize(("make_config", "resident"),
                         [(_lora_config, True), (lambda: _config(TrainingMethod.LORA), False)],
                         ids=["token-trained", "plain-lora"])
def test_qwen3_stays_resident_exactly_when_a_token_is_trained(make_config, resident):
    # Training a token turns the text cache off, so Qwen3 encodes live every step. If it is left evicted
    # to temp_device -- which is what latent_caching alone would decide -- every step either crashes on a
    # device mismatch or drags the encode through the offload path. The other direction costs VRAM for
    # nothing on a plain LoRA run, so both directions are pinned.
    config = make_config()
    config.latent_caching = True
    model, setup = _prepared(config)

    materialized = []
    model.materialize_only = lambda *parts: materialized.extend(parts)
    setup.setup_train_device(model, config)

    assert ("text_encoder" in materialized) is resident, \
        f"text_encoder residency is wrong for this configuration: {materialized}"


def test_norm_preservation_pulls_the_token_back_to_the_vocab_norm():
    config = _config(TrainingMethod.EMBEDDING)
    config.preserve_embedding_norm = True
    model, setup = _prepared(config)

    vector = _trained_embedding(model).vector
    with torch.no_grad():
        vector.mul_(10.0)

    setup.after_optimizer_step(model, config, TrainProgress())

    expected = model.embedding_wrapper.orig_median_norm
    assert torch.allclose(vector.detach().norm(dim=-1),
                          torch.full((vector.shape[0],), float(expected)), atol=1e-4), \
        "preserve_embedding_norm did not renormalize the trained rows"
