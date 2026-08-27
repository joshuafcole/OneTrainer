"""Prompt-pair Concept-Sliders training for Anima.

A slider is a plain LoRA/LoKr adapter whose signed multiplier is the control
knob, so this reuses AnimaLoRASetup's adapter construction wholesale and replaces
only the training step: instead of one forward against a dataset image, it runs
the velocity-space Concept-Sliders objective

    v*(x_t, c_t, t) = v(c_t) + eta * mean_p( v(c+,p) - v(c-,p) )

There is no image dataset (SliderPromptPairDataLoader emits step-driver batches),
so predict() generates an on-manifold x_t by partial flow-matching denoising
under the neutral target conditioning, then hands the multi-forward orchestration
to ModelSetupSliderMixin.

predict() returns the loss directly: the objective needs several forwards with
the adapter toggled, which does not fit the single-forward predict/calculate_loss
split. calculate_loss just unwraps it.
"""

from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupSliderMixin import ModelSetupSliderMixin
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.slider_util import alias_lora_persistence_to_slider
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

VAE_SCALE_FACTOR = 8

# Where the SDEdit trajectory starts. Anima's flow-matching sigma runs 1 (noise)
# to 0 (image); starting just under 1 keeps the first Euler step finite.
_TRAJECTORY_START_SIGMA = 0.999


@factory.register(BaseModelSetup, ModelType.ANIMA, TrainingMethod.SLIDER)
class AnimaSliderSetup(
    AnimaLoRASetup,
    ModelSetupSliderMixin,
):
    def setup_model(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ):
        # Nothing to add to AnimaLoRASetup's adapter construction -- Anima builds
        # only a transformer adapter and already refuses embeddings, so there is
        # no text-encoder adapter that could sit in the optimizer without ever
        # receiving gradient (contrast StableDiffusionXLSliderSetup, which has to
        # suppress two). The one thing worth establishing here is that the chosen
        # PEFT type has a multiplier to slide -- setup_model is the first per-run
        # hook a setup gets, so this is as early as the check can land.
        self._check_slider_peft_type(config)
        super().setup_model(model, config)

    def setup_train_device(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ):
        # A prompt-pair slider encodes its prompts live (then caches them in
        # process), so the text encoder must be resident regardless of the
        # latent_caching flag -- unlike AnimaLoRASetup, which can evict it once the
        # dataset is cached. The VAE is unused while training (x_t is synthetic)
        # but stays resident so sampling-during-training can decode without a
        # device shuffle. materialize_only also brings the text_conditioner along
        # with the text encoder; it is not in ModelType.model_parts().
        model.materialize_only("transformer", "text_encoder", "vae")

        model.text_encoder.eval()
        model.text_conditioner.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    # ---- conditioning -------------------------------------------------------

    def _encode(self, model: AnimaModel, text: str) -> Tensor:
        def encode(prompt: str) -> Tensor:
            # Frozen path: detach so nothing tries to backprop into the encoder.
            return model.encode_text(train_device=self.train_device, text=prompt).detach()

        return self._slider_cached_conditioning(text, encode)

    # ---- x_t generation -----------------------------------------------------

    @torch.no_grad()
    def _build_xt(self, model, eh_target, sigma, shape, padding_mask, dtype, gen, set_multiplier, anchor_steps):
        """The noised latent for this slider step.

        anchor_steps > 0: Euler-integrate the flow-matching ODE (dx/dsigma = v)
        from ~pure noise down to `sigma` under the neutral target conditioning with
        the adapter OFF, so x_t lands on the base model's own trajectory (SDEdit).
        anchor_steps == 0: plain Gaussian noise -- cheaper, but off-manifold, so the
        base's guidance direction there is measured somewhere the model never
        actually visits.
        """
        set_multiplier(0.0)
        x = torch.randn(shape, generator=gen).to(self.train_device, dtype=dtype)
        if anchor_steps <= 0:
            return x
        sigmas = torch.linspace(_TRAJECTORY_START_SIGMA, float(sigma), anchor_steps + 1)
        for i in range(anchor_steps):
            s = sigmas[i].item()
            t_norm = torch.full((1,), s, device=self.train_device, dtype=dtype)
            v = model.transformer(
                hidden_states=x,
                timestep=t_norm,
                encoder_hidden_states=eh_target.to(dtype=dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            x = x + (sigmas[i + 1].item() - s) * v
        return x

    # ---- training step ------------------------------------------------------

    def predict(
        self,
        model: AnimaModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        if config.slider_regime != SliderRegime.PROMPT_PAIR:
            raise NotImplementedError(f"slider regime {config.slider_regime} is not implemented")

        triples = self._slider_triples(config)
        wrapper = model.transformer_lora
        dtype = model.train_dtype.torch_dtype()

        with model.autocast_context:
            seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            rand = Random(seed)
            gen = torch.Generator(device="cpu").manual_seed(seed)

            triple = self._choose_triple(triples, rand)
            eh_target = self._encode(model, triple.target)
            positive_texts, negative_texts = self._slider_prompt_pairs(triple, config)
            positive_conds = [self._encode(model, text) for text in positive_texts]
            negative_conds = [self._encode(model, text) for text in negative_texts]

            h_pix, w_pix = self._slider_resolution(config)
            h_lat, w_lat = h_pix // VAE_SCALE_FACTOR, w_pix // VAE_SCALE_FACTOR
            in_ch = model.transformer.config.in_channels
            # Anima latents are 5D (B,C,T=1,H,W); the Cosmos transformer wants a
            # pixel-space padding mask.
            shape = (1, in_ch, 1, h_lat, w_lat)
            padding_mask = torch.zeros(
                (1, 1, h_lat * VAE_SCALE_FACTOR, w_lat * VAE_SCALE_FACTOR),
                device=self.train_device, dtype=dtype,
            )

            sigma = self._slider_sample_noise_level(config, rand)
            t_norm = torch.full((1,), sigma, device=self.train_device, dtype=dtype)

            x_t = self._build_xt(
                model, eh_target, sigma, shape, padding_mask, dtype, gen,
                wrapper.set_multiplier, int(config.slider_anchor_steps),
            )

            def run_velocity(conditioning: Tensor) -> Tensor:
                return model.transformer(
                    hidden_states=x_t,
                    timestep=t_norm,
                    encoder_hidden_states=conditioning.to(dtype=dtype),
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

            loss = self._slider_prompt_loss(
                run_velocity,
                wrapper.set_multiplier,
                eh_target,
                positive_conds,
                negative_conds,
                eta=float(config.slider_eta),
                strength=float(config.slider_strength),
                symmetric=bool(config.slider_symmetric),
            )

        return {"loss": loss}

    def calculate_loss(self, model: AnimaModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return data["loss"]


alias_lora_persistence_to_slider(ModelType.ANIMA)
