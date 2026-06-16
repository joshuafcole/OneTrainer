"""AnimaSliderSetup -- prompt-pair Concept-Sliders training for Anima.

A slider is a plain LoRA/LoKr adapter whose signed multiplier is a control knob,
so this reuses AnimaLoRASetup's adapter construction wholesale and only replaces
the training step: instead of a single forward against a dataset image, it runs
the velocity-space Concept-Sliders objective (docs/slider_lora.md S2)

    v*(x_t, c_t, t) = v(c_t) + eta * mean_p( v(c+,p) - v(c-,p) )

The frozen base supplies the guidance direction; only the adapter trains. There
is no image dataset (AnimaSliderDataLoader emits empty step-driver batches), so
predict() generates an on-manifold x_t by partial flow-matching denoising under
the neutral target conditioning (SDEdit), then delegates the multi-forward
orchestration to ModelSetupSliderMixin.

predict() returns the loss directly under model.autocast_context (the objective
needs several forwards with the adapter toggled, which does not fit the
single-forward predict/calculate_loss split); calculate_loss just unwraps it.
"""

from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupSliderMixin import ModelSetupSliderMixin
from modules.util import factory
from modules.util.config.SliderConfig import SliderPromptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

VAE_SCALE_FACTOR = 8


class AnimaSliderSetup(
    AnimaLoRASetup,
    ModelSetupSliderMixin,
):
    def __init__(self, train_device, temp_device, debug_mode):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)
        # encode_text output (the Cosmos cross-attention conditioning) cached per
        # prompt string: the slider prompts are fixed across the whole run, so we
        # never re-run Qwen3 + the conditioner for a prompt twice.
        self._cond_cache: dict[str, Tensor] = {}

    def setup_train_device(self, model: AnimaModel, config: TrainConfig):
        # Prompt-pair sliders have no latent/text cache: prompts are encoded live
        # (then cached in-process), so Qwen3 + the conditioner must be resident
        # regardless of the latent_caching flag. The VAE is unused while training
        # (x_t is synthetic) but is kept resident so sampling-during-training can
        # decode without a device shuffle.
        model.text_encoder_to(self.train_device)
        model.text_conditioner_to(self.train_device)
        model.vae_to(self.train_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.text_conditioner.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    # ---- prompt / conditioning helpers -------------------------------------

    def _encode_cached(self, model: AnimaModel, text: str) -> Tensor:
        cached = self._cond_cache.get(text)
        if cached is None:
            # Frozen path: detach so nothing tries to backprop into the encoder.
            cached = model.encode_text(train_device=self.train_device, text=text).detach()
            self._cond_cache[text] = cached
        return cached

    def _build_pairs(self, model: AnimaModel, triple: SliderPromptConfig, config: TrainConfig):
        """Return (positive_conds, negative_conds) for the guidance direction.

        Bare pair (CS Eq. 7) when no preservation set is configured; otherwise
        the attribute poles are re-stated in each preservation context and the
        mixin averages the per-context delta (the disentanglement mean, CS Eq. 8).
        """
        contexts = [p.strip() for p in config.slider_preservation_prompts.split("|") if p.strip()]
        if not contexts:
            return [self._encode_cached(model, triple.positive)], [self._encode_cached(model, triple.negative)]

        positive, negative = [], []
        for ctx in [None, *contexts]:  # always include the bare pair plus each context
            p = triple.positive if ctx is None else f"{triple.positive}, {ctx}"
            n = triple.negative if ctx is None else f"{triple.negative}, {ctx}"
            positive.append(self._encode_cached(model, p))
            negative.append(self._encode_cached(model, n))
        return positive, negative

    def _choose_triple(self, triples: list[SliderPromptConfig], rand: Random) -> SliderPromptConfig:
        weights = [max(0.0, t.weight) for t in triples]
        total = sum(weights)
        if total <= 0.0:
            return rand.choice(triples)
        r = rand.random() * total
        acc = 0.0
        for triple, w in zip(triples, weights, strict=True):
            acc += w
            if r <= acc:
                return triple
        return triples[-1]

    def _latent_shape(self, model: AnimaModel, config: TrainConfig) -> tuple[int, int, int]:
        token = config.resolution.split(",")[0].strip().lower()
        if "x" in token:
            h_str, w_str = token.split("x", 1)
            h_pix, w_pix = int(h_str), int(w_str)
        else:
            h_pix = w_pix = int(token)
        in_ch = int(getattr(getattr(model.transformer, "config", None), "in_channels", 16) or 16)
        return in_ch, h_pix // VAE_SCALE_FACTOR, w_pix // VAE_SCALE_FACTOR

    # ---- x_t generation -----------------------------------------------------

    @torch.no_grad()
    def _build_xt(self, model, eh_target, sigma, shape, padding_mask, dtype, gen, set_multiplier, anchor_steps):
        """Synthetic noised latent for the slider step.

        anchor_steps > 0: Euler-integrate the conditional flow-matching ODE
        (dx/dsigma = v, v = noise - image) from ~noise down to ``sigma`` under the
        neutral target conditioning, with the adapter OFF, so x_t lands on the
        base model's own trajectory (SDEdit). anchor_steps == 0: plain Gaussian
        noise (off-manifold; cheaper but the probe showed it under-measures).
        """
        set_multiplier(0.0)
        x = torch.randn(shape, generator=gen).to(self.train_device, dtype=dtype)
        if anchor_steps <= 0:
            return x
        sigmas = torch.linspace(0.999, float(sigma), anchor_steps + 1)
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
            raise NotImplementedError(
                f"slider regime {config.slider_regime} is not wired for Anima yet (prompt-pair only)"
            )

        triples = [t for t in config.slider_prompts if t.enabled]
        if not triples:
            raise RuntimeError("Slider training needs at least one enabled prompt pair (see the Slider tab).")

        wrapper = model.transformer_lora
        dtype = model.train_dtype.torch_dtype()

        with model.autocast_context:
            seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            rand = Random(seed)
            gen = torch.Generator(device="cpu").manual_seed(seed)

            triple = self._choose_triple(triples, rand)
            eh_target = self._encode_cached(model, triple.target)
            positive_conds, negative_conds = self._build_pairs(model, triple, config)

            in_ch, h_lat, w_lat = self._latent_shape(model, config)
            shape = (1, in_ch, 1, h_lat, w_lat)
            h_pix, w_pix = h_lat * VAE_SCALE_FACTOR, w_lat * VAE_SCALE_FACTOR
            padding_mask = torch.zeros((1, 1, h_pix, w_pix), device=self.train_device, dtype=dtype)

            lo, hi = float(config.slider_sigma_min), float(config.slider_sigma_max)
            if hi < lo:
                lo, hi = hi, lo
            sigma = lo + (hi - lo) * rand.random()
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


factory.register(BaseModelSetup, AnimaSliderSetup, ModelType.ANIMA, TrainingMethod.SLIDER)

# A slider IS a LoRA file: reuse the generic Anima LoRA saver/loader (already
# registered for (ANIMA, LORA), since modelSaver/modelLoader are imported before
# modelSetup). create_model_saver/loader do not fall back to model-type-only, so
# SLIDER must be registered explicitly. The sampler does fall back, so it needs
# no SLIDER entry.
_lora_saver = factory.get(BaseModelSaver, ModelType.ANIMA, TrainingMethod.LORA)
if _lora_saver is not None:
    factory.register(BaseModelSaver, _lora_saver, ModelType.ANIMA, TrainingMethod.SLIDER)
_lora_loader = factory.get(BaseModelLoader, ModelType.ANIMA, TrainingMethod.LORA)
if _lora_loader is not None:
    factory.register(BaseModelLoader, _lora_loader, ModelType.ANIMA, TrainingMethod.SLIDER)
