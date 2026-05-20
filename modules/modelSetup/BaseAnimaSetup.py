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
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
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
        "detail": {'patterns': ["^(?=.*attn)(?!.*norm).*",
                                "^(?=.*ff\\.net).*",
                                "^(?=.*norm[123]\\.linear).*"], 'regex': True},
        "attn-mlp": {'patterns': ["^(?=.*attn)(?!.*norm).*",
                                  "^(?=.*ff\\.net).*"], 'regex': True},
        "attn-only": {'patterns': ["^(?=.*attn)(?!.*norm).*"], 'regex': True},
        "cross-attn": {'patterns': ["^(?=.*attn2)(?!.*norm).*"], 'regex': True},
    }

    def setup_optimizations(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ):
        # CosmosTransformer3DModel supports gradient checkpointing via the
        # standard PeftAdapterMixin method. No specialized offload
        # conductor (yet) -- the layerwise-offload integration can be
        # added by a future patch following the
        # enable_checkpointing_for_z_image_transformer pattern.
        if config.gradient_checkpointing.enabled():
            if hasattr(model.transformer, "enable_gradient_checkpointing"):
                model.transformer.enable_gradient_checkpointing()
            if model.text_encoder is not None and \
                    hasattr(model.text_encoder, "gradient_checkpointing_enable"):
                model.text_encoder.gradient_checkpointing_enable()

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        # Text encoder + conditioner train dtype mirrors the transformer.
        # AnimaTextConditioner is frozen during LoRA training, so we
        # don't need a separate disable_fp16_autocast_context wrapper.
        model.text_encoder_train_dtype = model.train_dtype

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.text_conditioner, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

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
            # Stage B caching: the dataloader cached qwen_hidden_state and
            # both sets of t5 tokens; encode_text skips Qwen3 and runs only
            # AnimaTextConditioner here. encoder_hidden_states is the
            # (B, T_t5, 1024) tensor the Cosmos cross-attention consumes.
            encoder_hidden_states = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                qwen_hidden_states=batch.get('text_encoder_hidden_state'),
                tokens_mask_qwen=batch['tokens_mask_qwen'],
                tokens_t5=batch['tokens_t5'],
                tokens_mask_t5=batch['tokens_mask_t5'],
                text_encoder_dropout_probability=None,
            )

            # ---- latents ---------------------------------------------------
            # vae_frame_dim=True in the dataloader gives us 5D
            # (B, C, T=1, H_lat, W_lat); defend against 4D in case a future
            # caching path strips T.
            latent_image = batch['latent_image']
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)

            scaled_latent_image = model.scale_latents(latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            # Anima's scheduler has a static shift=3.0 already; we pull
            # the value from config.timestep_shift so the user can adjust
            # via the training UI without touching the scheduler.
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=config.timestep_shift,
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
                timestep.to(dtype=model.train_dtype.torch_dtype())
                / model.noise_scheduler.config.num_train_timesteps
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
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens_qwen'], model.tokenizer, config, train_progress)
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
        model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
