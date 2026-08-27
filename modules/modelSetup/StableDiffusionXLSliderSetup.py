"""Prompt-pair Concept-Sliders training for SDXL.

The second host for ModelSetupSliderMixin, and the one that makes
"model-agnostic" checkable rather than asserted: Anima is a flow-matching
transformer with a single fused conditioning tensor, SDXL is an epsilon (or
v-prediction) UNet with a two-encoder conditioning plus SDXL's micro-conditioning
time ids. The objective is unchanged between them -- everything that differs
lives in `run_velocity` and `_build_xt` below.

Why an epsilon model needs no separate objective: at a fixed (x_t, t) the
velocity and the noise prediction are related by v = a(t)*x_t + b(t)*eps, so
eps(c+) - eps(c-) is the same direction as v(c+) - v(c-) up to the positive
factor b(t). See ModelSetupSliderMixin's docstring.
"""

from random import Random
from typing import NamedTuple

import modules.util.multi_gpu_util as multi
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupSliderMixin import ModelSetupSliderMixin
from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.slider_util import alias_lora_persistence_to_slider
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

VAE_SCALE_FACTOR = 8


class SliderConditioning(NamedTuple):
    """One encoded prompt, in the shape SDXL's UNet consumes.

    Both halves vary with the prompt -- the pooled embedding is not shared -- so
    they travel together as the objective's opaque conditioning value.
    """

    prompt_embeds: Tensor
    pooled_embeds: Tensor


@factory.register(BaseModelSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE, TrainingMethod.SLIDER)
@factory.register(BaseModelSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING, TrainingMethod.SLIDER)
class StableDiffusionXLSliderSetup(
    StableDiffusionXLLoRASetup,
    ModelSetupSliderMixin,
):
    # ---- adapter construction ----------------------------------------------
    #
    # A slider trains the denoiser adapter and nothing else. Its conditionings are
    # encoded once per run and detached -- they are fixed for the whole run, and
    # the frozen base's guidance direction must not be a gradient path -- so no
    # gradient can reach a text-encoder adapter or a trained embedding. Building
    # them anyway would put parameters in the optimizer that provably never move
    # and write them into the output file.
    #
    # `text_encoder.train` defaults to on, so a user reaching this from LoRA
    # training has not asked for anything: warn and carry on. Embedding training
    # takes several deliberate steps to switch on, so refuse it rather than
    # quietly do something else.

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "unet_lora", model.unet_lora, config.unet)
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad(
            "unet_lora", model.unet_lora, config.unet, model.train_progress)

    @staticmethod
    def _check_trainable_parts(config: TrainConfig) -> None:
        """The policy above, as one checkable step."""
        if config.train_any_embedding() or config.train_any_output_embedding():
            raise RuntimeError(
                "Slider training cannot train embeddings: the slider prompts are encoded "
                "once and frozen, so an embedding would receive no gradient. Turn off "
                "embedding training, or use the Embedding training method instead."
            )
        if config.text_encoder.train or config.text_encoder_2.train:
            print(
                "Warning: slider training does not train the text encoders (the slider "
                "prompts are encoded once and frozen), so no text encoder adapter was "
                "created. Only the UNet adapter will train."
            )

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        self._check_slider_peft_type(config)
        self._check_trainable_parts(config)

        model.text_encoder_1_lora = None
        model.text_encoder_2_lora = None
        model.unet_lora = LoRAModuleWrapper(
            model.unet, "unet", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.unet_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.unet_lora.set_dropout(config.dropout_probability)
        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.hook_to_module()

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        # Both text encoders stay resident: a slider encodes its prompts live on the
        # first step regardless of latent_caching (there is no dataset to cache).
        # The VAE is unused -- x_t is synthetic -- but stays resident so
        # sampling-during-training can decode without a device shuffle.
        model.materialize_only("unet", "text_encoder", "text_encoder_2", "vae")

        model.text_encoder_1.eval()
        model.text_encoder_2.eval()
        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    # ---- conditioning -------------------------------------------------------

    def _encode(self, model: StableDiffusionXLModel, text: str) -> SliderConditioning:
        def encode(prompt: str) -> SliderConditioning:
            prompt_embeds, pooled_embeds = model.combine_text_encoder_output(*model.encode_text(
                train_device=self.train_device,
                batch_size=1,
                text=prompt,
            ))
            # Frozen path: detach so nothing tries to backprop into the encoders.
            return SliderConditioning(prompt_embeds.detach(), pooled_embeds.detach())

        return self._slider_cached_conditioning(text, encode)

    # ---- x_t generation -----------------------------------------------------

    def _coefficients(self, model: StableDiffusionXLModel) -> DiffusionScheduleCoefficients:
        cached = getattr(self, "_slider_coefficients", None)
        if cached is None:
            cached = DiffusionScheduleCoefficients.from_betas(
                model.noise_scheduler.betas.to(device=self.train_device)
            )
            self._slider_coefficients = cached
        return cached

    def _to_epsilon(self, model, prediction: Tensor, x: Tensor, timestep: int) -> Tensor:
        """The UNet's output as a noise prediction.

        A v-prediction model -- which every zero-terminal-SNR-rescaled SDXL is,
        since rescale_noise_scheduler_to_zero_terminal_snr() forces it -- returns
        velocity, and the DDIM step below is written in epsilon. The objective
        itself is indifferent (see the module docstring); only the anchor
        trajectory needs the conversion.
        """
        if model.noise_scheduler.config.prediction_type != 'v_prediction':
            return prediction
        coefficients = self._coefficients(model)
        sqrt_alpha = coefficients.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = coefficients.sqrt_one_minus_alphas_cumprod[timestep]
        return sqrt_alpha * prediction + sqrt_one_minus_alpha * x

    @torch.no_grad()
    def _build_xt(self, model, conditioning, timestep, shape, add_time_ids, dtype, gen,
                  set_multiplier, anchor_steps):
        """The noised latent for this slider step.

        anchor_steps > 0: DDIM-sample from pure noise down to `timestep` under the
        neutral target conditioning with the adapter OFF, so x_t lands on the base
        model's own trajectory (SDEdit). anchor_steps == 0: plain Gaussian noise --
        cheaper, but off-manifold, so the guidance direction is measured somewhere
        the model never actually visits.
        """
        set_multiplier(0.0)
        x = torch.randn(shape, generator=gen).to(self.train_device, dtype=dtype)
        if anchor_steps <= 0:
            return x

        coefficients = self._coefficients(model)
        num_timesteps = model.noise_scheduler.config['num_train_timesteps']
        schedule = torch.linspace(num_timesteps - 1, timestep, anchor_steps + 1).round().long().tolist()

        for i in range(anchor_steps):
            t_now, t_next = schedule[i], schedule[i + 1]
            predicted = model.unet(
                sample=x,
                timestep=torch.full((1,), t_now, device=self.train_device, dtype=torch.long),
                encoder_hidden_states=conditioning.prompt_embeds.to(dtype=dtype),
                added_cond_kwargs={
                    "text_embeds": conditioning.pooled_embeds.to(dtype=dtype),
                    "time_ids": add_time_ids,
                },
            ).sample
            epsilon = self._to_epsilon(model, predicted, x, t_now)
            x0 = (x - coefficients.sqrt_one_minus_alphas_cumprod[t_now] * epsilon) \
                / coefficients.sqrt_alphas_cumprod[t_now]
            x = coefficients.sqrt_alphas_cumprod[t_next] * x0 \
                + coefficients.sqrt_one_minus_alphas_cumprod[t_next] * epsilon
            x = x.to(dtype=dtype)
        return x

    # ---- training step ------------------------------------------------------

    def predict(
        self,
        model: StableDiffusionXLModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        if config.slider_regime != SliderRegime.PROMPT_PAIR:
            raise NotImplementedError(f"slider regime {config.slider_regime} is not implemented")

        triples = self._slider_triples(config)
        wrapper = model.unet_lora
        dtype = model.train_dtype.torch_dtype()

        with model.autocast_context:
            seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            rand = Random(seed)
            gen = torch.Generator(device="cpu").manual_seed(seed)

            triple = self._choose_triple(triples, rand)
            target_cond = self._encode(model, triple.target)
            positive_texts, negative_texts = self._slider_prompt_pairs(triple, config)
            positive_conds = [self._encode(model, text) for text in positive_texts]
            negative_conds = [self._encode(model, text) for text in negative_texts]

            h_pix, w_pix = self._slider_resolution(config)
            shape = (1, model.unet.config.in_channels, h_pix // VAE_SCALE_FACTOR, w_pix // VAE_SCALE_FACTOR)
            # SDXL micro-conditioning: original size, crop offset, target size. A
            # synthetic sample has no crop and no original beyond what we asked for.
            add_time_ids = torch.tensor(
                [[h_pix, w_pix, 0, 0, h_pix, w_pix]], device=self.train_device, dtype=dtype,
            )

            num_timesteps = model.noise_scheduler.config['num_train_timesteps']
            noise_level = self._slider_sample_noise_level(config, rand)
            timestep_index = int(round(noise_level * (num_timesteps - 1)))
            timestep = torch.full((1,), timestep_index, device=self.train_device, dtype=torch.long)

            x_t = self._build_xt(
                model, target_cond, timestep_index, shape, add_time_ids, dtype, gen,
                wrapper.set_multiplier, int(config.slider_anchor_steps),
            )

            def run_velocity(conditioning: SliderConditioning) -> Tensor:
                return model.unet(
                    sample=x_t,
                    timestep=timestep,
                    encoder_hidden_states=conditioning.prompt_embeds.to(dtype=dtype),
                    added_cond_kwargs={
                        "text_embeds": conditioning.pooled_embeds.to(dtype=dtype),
                        "time_ids": add_time_ids,
                    },
                ).sample

            loss = self._slider_prompt_loss(
                run_velocity,
                wrapper.set_multiplier,
                target_cond,
                positive_conds,
                negative_conds,
                eta=float(config.slider_eta),
                strength=float(config.slider_strength),
                symmetric=bool(config.slider_symmetric),
            )

        return {"loss": loss}

    def calculate_loss(self, model: StableDiffusionXLModel, batch: dict, data: dict,
                       config: TrainConfig) -> Tensor:
        return data["loss"]


alias_lora_persistence_to_slider(
    ModelType.STABLE_DIFFUSION_XL_10_BASE,
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)
