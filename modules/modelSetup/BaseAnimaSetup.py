"""BaseAnimaSetup -- training-step machinery for Anima.

Differences from BaseZImageSetup that matter:

  - The Cosmos transformer takes ``(B, C, T=1, H, W)`` latents +
    a pixel-resolution ``padding_mask``, and returns velocity in
    the (noise - image) direction directly. No timestep inversion
    (``timestep / num_train_timesteps``, not 1 - ...) and no sign
    flip on the predicted flow.

  - Text conditioning is two-stage: ``model.encode_text`` consumes
    the cached Qwen3 last_hidden_state plus T5 token ids and runs
    AnimaTextConditioner on the train device every step. The
    conditioner is frozen but it is NOT cached -- caching it would
    double text-cache footprint for a deterministic 10 ms op.
"""

from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.AnimaModel import AnimaModel, AnimaModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_cosmos_transformer,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

VAE_SCALE_FACTOR = 8  # AutoencoderKLQwenImage spatial compression ratio (2^3)


class BaseAnimaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    # LoRA layer presets for the Cosmos transformer. Anatomy from
    # CosmosTransformerBlock: each block has attn1 (self), attn2 (cross),
    # ff (feedforward), and three adaln modulation blocks. Cosmos puts
    # everything under transformer_blocks.<i>.<part>.
    #
    # detail is the broadest LoRA surface short of `blocks`/`full`: it
    # adds the adaln modulations (norm1/2/3.linear_1/2) on top of
    # attn-mlp so style-heavy LoRAs have access to the per-block
    # conditioning offsets. Roughly 1.4x the parameter count of
    # attn-mlp at the same rank.
    LAYER_PRESETS = {
        "full": [],
        "blocks": ["transformer_blocks"],
        "detail": {
            "patterns": ["^(?=.*attn)(?!.*norm).*", "^(?=.*ff\\.net).*", "^(?=.*norm[123]\\.linear).*"],
            "regex": True,
        },
        "attn-mlp": {"patterns": ["^(?=.*attn)(?!.*norm).*", "^(?=.*ff\\.net).*"], "regex": True},
        "attn-only": {"patterns": ["^(?=.*attn)(?!.*norm).*"], "regex": True},
        "cross-attn": {"patterns": ["^(?=.*attn2)(?!.*norm).*"], "regex": True},
    }

    def setup_optimizations(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        # Gradient checkpointing goes through OneTrainer's conductor-based
        # wrappers (not the diffusers built-in) so that CPU_OFFLOADED can
        # stream the frozen Cosmos transformer's weights -- and optionally
        # its activations -- to system RAM via the LayerOffloadConductor.
        # Plain ON leaves the conductor inert (recompute only); offloading
        # only activates when gradient_checkpointing.offload() is set and
        # layer_offload_fraction > 0 (weights) / enable_activation_offloading
        # (activations). The conductor's checkpoint wrapper also handles
        # blocks with no trainable params, which is the LoRA/TI case where
        # the transformer is frozen but gradient still flows through it.
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = enable_checkpointing_for_cosmos_transformer(model.transformer, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(
                    model.text_encoder, config
                )

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                config.weight_dtypes().embedding if config.train_any_embedding() else None,
            ],
            config.enable_autocast_cache,
        )

        # Text encoder + conditioner train dtype mirrors the transformer.
        # AnimaTextConditioner is frozen during LoRA training, so we
        # don't need a separate disable_fp16_autocast_context wrapper.
        model.text_encoder_train_dtype = model.train_dtype

        # When training a TI embedding, Qwen3 is run live every step (its
        # cached hidden states are disabled so gradients can reach the
        # trainable token vectors), so it needs a real autocast context.
        # The cached path never runs Qwen3 at step time and leaves this as
        # the default nullcontext.
        if config.train_any_embedding():
            model.text_encoder_autocast_context, model.text_encoder_train_dtype = create_autocast_context(
                self.train_device,
                config.train_dtype,
                [
                    config.weight_dtypes().text_encoder,
                    config.weight_dtypes().embedding,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.text_conditioner, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        # TI tokens are injected into both the Qwen3 word embedding table
        # and the T5 input embedding table (inside the frozen
        # AnimaTextConditioner). Output embeddings aren't supported for
        # Anima yet (no create_output_embedding_fn), which is fine --
        # additional embeddings used alongside LoRA are always input
        # embeddings.
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            saved_state = model.embedding_state_dicts.get(embedding_config.uuid, None)

            # Qwen3 vector: seed from the host vocab via the configured
            # initial_embedding_text, or load from checkpoint.
            qwen_state = None
            if saved_state is not None:
                qwen_state = saved_state.get("qwen_out", saved_state.get("qwen", None))
            if qwen_state is None:
                with model.autocast_context:
                    qwen_state = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer,
                        model.text_encoder,
                    )
            if qwen_state is not None:
                qwen_state = qwen_state.to(
                    dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                    device=self.train_device,
                ).detach()

            # T5 vector: only built when T5-side training is enabled. The
            # default (config.train_t5_embedding off) leaves T5 untouched so
            # the trained concept is reproducible in ComfyUI, whose Anima
            # encoder doesn't inject T5-side embeddings -- the placeholder is
            # left to tokenize naturally on the T5 side. When enabled, there
            # is no host-vocab seeding (the conditioner wasn't trained to
            # consume placeholder strings on the T5 side): load from
            # checkpoint when resuming, otherwise init with small Gaussian
            # noise rescaled to the median T5 row norm.
            t5_state = None
            if config.train_t5_embedding and model.text_conditioner is not None:
                if saved_state is not None:
                    t5_state = saved_state.get("t5", None)
                if t5_state is None and embedding_config.token_count is not None:
                    t5_state = self._create_noise_embedding(
                        host_embedding=model.text_conditioner.embed,
                        token_count=embedding_config.token_count,
                        dtype=model.text_conditioner.embed.weight.dtype,
                        device=self.train_device,
                    )
            if t5_state is not None:
                t5_state = t5_state.to(
                    dtype=model.text_conditioner.embed.weight.dtype,
                    device=self.train_device,
                ).detach()

            embedding = AnimaModelEmbedding(
                embedding_config.uuid,
                qwen_state,
                t5_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        if model.tokenizer is not None:
            self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())
        # Only register T5 placeholder tokens when T5-side training is on.
        # Off (the default) leaves the placeholder to tokenize naturally,
        # matching ComfyUI's Anima encoder.
        if model.t5_tokenizer is not None and config.train_t5_embedding:
            self._add_embeddings_to_tokenizer(model.t5_tokenizer, model.all_t5_embeddings())

    def _setup_embedding_wrapper(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        if model.tokenizer is not None and model.text_encoder is not None:
            model.embedding_wrapper = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer,
                orig_module=model.text_encoder.get_input_embeddings(),
                embeddings=model.all_text_encoder_embeddings(),
            )

        if model.embedding_wrapper is not None:
            model.embedding_wrapper.hook_to_module()

        # The T5 wrapper hooks trainable rows into the conditioner's embed
        # table. Skip it entirely when T5-side training is off (the default):
        # there are no T5 vectors to splice in, and leaving the table
        # unhooked is what makes the placeholder tokenize naturally for
        # ComfyUI parity.
        if model.t5_tokenizer is not None and model.text_conditioner is not None and config.train_t5_embedding:
            model.t5_embedding_wrapper = AdditionalEmbeddingWrapper(
                tokenizer=model.t5_tokenizer,
                orig_module=model.text_conditioner.embed,
                embeddings=model.all_t5_embeddings(),
            )

        if model.t5_embedding_wrapper is not None:
            model.t5_embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        if model.text_encoder is not None:
            for embedding, embedding_config in zip(
                model.all_text_encoder_embeddings(), config.all_embedding_configs(), strict=True
            ):
                train_embedding = (
                    embedding_config.train
                    and config.text_encoder.train_embedding
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                )
                embedding.requires_grad_(train_embedding)
        # T5-side vectors only exist when train_t5_embedding is on; otherwise
        # they're None (no injection) and there is nothing to flag.
        if model.text_conditioner is not None and config.train_t5_embedding:
            for embedding, embedding_config in zip(
                model.all_t5_embeddings(), config.all_embedding_configs(), strict=True
            ):
                train_embedding = (
                    embedding_config.train
                    and config.text_encoder.train_embedding
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                )
                embedding.requires_grad_(train_embedding)

    def predict(
        self,
        model: AnimaModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            # ---- text ------------------------------------------------------
            # Normal (no-embedding) path -- Stage B caching: the dataloader
            # cached qwen_hidden_state + both sets of t5 tokens; encode_text
            # skips Qwen3 and runs only AnimaTextConditioner here.
            #
            # TI-embedding path: the text cache is disabled, so
            # text_encoder_hidden_state is absent and we pass tokens_qwen
            # instead -- encode_text then runs Qwen3 live (through the
            # embedding wrapper) so gradients reach the trainable token
            # vectors. encoder_hidden_states is the (B, T_t5, 1024) tensor
            # the Cosmos cross-attention consumes.
            encoder_hidden_states = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                tokens_qwen=batch["tokens_qwen"],
                qwen_hidden_states=batch.get("text_encoder_hidden_state")
                if not config.train_text_encoder_or_embedding()
                else None,
                tokens_mask_qwen=batch["tokens_mask_qwen"],
                tokens_t5=batch["tokens_t5"],
                tokens_mask_t5=batch["tokens_mask_t5"],
                text_encoder_dropout_probability=None,
                text_encoder_sequence_length=config.text_encoder_sequence_length,
            )

            # ---- latents ---------------------------------------------------
            # vae_frame_dim=True in the dataloader gives us 5D
            # (B, C, T=1, H_lat, W_lat); defend against 4D in case a future
            # caching path strips T.
            latent_image = batch["latent_image"]
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)

            scaled_latent_image = model.scale_latents(latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            # When dynamic_timestep_shifting is on, compute the shift
            # from the image's latent sequence length using Cosmos's
            # exponential mu mapping (see AnimaModel.calculate_timestep_shift).
            # Otherwise fall back to the static config.timestep_shift (3.0
            # by default, matching the scheduler's own shift field).
            if config.dynamic_timestep_shifting:
                shift = model.calculate_timestep_shift(
                    scaled_latent_image.shape[-2],
                    scaled_latent_image.shape[-1],
                )
            else:
                shift = config.timestep_shift
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config["num_train_timesteps"],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # ---- transformer ----------------------------------------------
            # Cosmos timestep is normalized t/num_train_timesteps (per
            # AnimaLoopBeforeDenoiser). padding_mask is at the *pixel*
            # resolution (vae_scale_factor * latent dims) -- the
            # transformer resizes it internally.
            t_norm = (
                timestep.to(dtype=model.train_dtype.torch_dtype()) / model.noise_scheduler.config.num_train_timesteps
            )
            latent_input = scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype())

            h_pix = scaled_latent_image.shape[-2] * VAE_SCALE_FACTOR
            w_pix = scaled_latent_image.shape[-1] * VAE_SCALE_FACTOR
            padding_mask = latent_input.new_zeros((1, 1, h_pix, w_pix))

            predicted_flow_5d = model.transformer(
                hidden_states=latent_input,
                timestep=t_norm,
                encoder_hidden_states=encoder_hidden_states.to(dtype=model.train_dtype.torch_dtype()),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            # Cosmos returns velocity in the (noise - image) direction
            # directly -- no sign flip, unlike Z-Image's transformer
            # which inverts it.
            predicted_flow = predicted_flow_5d

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_flow,
                "target": flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch["tokens_qwen"], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
        self,
        model: AnimaModel,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: AnimaModel, config: TrainConfig):
        # Pre-caching pass: only Qwen3 needs to be on the train device,
        # since we cache its last_hidden_state. The conditioner runs at
        # training step time and can stay on the temp device until then.
        model.to(self.temp_device)

        # When training a TI embedding the text cache is disabled (Qwen3
        # runs live each step under grad), so there is no text-caching pass
        # that needs Qwen3 on the train device.
        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
