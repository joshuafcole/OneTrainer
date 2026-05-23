"""Anima sampler.

Implements the inference loop translated from the Anima modular
pipeline blocks:

  - ``diffusers.modular_pipelines.anima.encoders.AnimaTextEncoderStep``
  - ``diffusers.modular_pipelines.anima.before_denoise``
  - ``diffusers.modular_pipelines.anima.denoise``
  - ``diffusers.modular_pipelines.anima.decoders``

We deliberately do NOT use AnimaModularPipeline at sample time. Going
flat lets us interleave per-stage device shuffling (Qwen3 +
AnimaTextConditioner -> transformer -> VAE), which is what every
other OneTrainer sampler does and what makes layerwise offload and
latent caching usable.
"""

from collections.abc import Callable

from modules.model.AnimaModel import AnimaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

import numpy as np
from tqdm import tqdm

# Matches the pipeline-side default that the AnimaAutoBlocks image_processor
# is configured with. Tied to the AutoencoderKLQwenImage four-stage dim_mult
# (2^3 = 8 spatial downsamples).
VAE_SCALE_FACTOR = 8


class AnimaSampler(BaseModelSampler):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        model: AnimaModel,
        model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        # AnimaModularPipeline carries an image_processor we reuse for the
        # final tensor -> PIL step; we never actually call __call__ on it.
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __encode_prompts(
        self,
        prompt: str,
        negative_prompt: str,
        cfg_enabled: bool,
        text_encoder_sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode positive and (optionally) negative prompts.

        Returns the AnimaTextConditioner outputs ready to feed into the
        Cosmos transformer's cross-attention. Two separate calls so
        that downstream CFG combines them with a single per-step
        transformer subtraction rather than running them through a
        batched conditional pass (cheaper memory, easier offload).
        """
        prompt_embeds = self.model.encode_text(
            text=prompt,
            batch_size=1,
            train_device=self.train_device,
            text_encoder_sequence_length=text_encoder_sequence_length,
        )

        negative_prompt_embeds = None
        if cfg_enabled:
            negative_prompt_embeds = self.model.encode_text(
                text=negative_prompt if negative_prompt is not None else "",
                batch_size=1,
                train_device=self.train_device,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __sample_base(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        seed: int,
        random_seed: bool,
        diffusion_steps: int,
        cfg_scale: float,
        noise_scheduler: NoiseScheduler,
        text_encoder_sequence_length: int | None = None,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # Use a fresh scheduler instance per call so multi-sample runs
            # don't share mutable scheduler state (begin_index, sigmas).
            import copy

            scheduler = copy.deepcopy(self.model.noise_scheduler)
            image_processor = self.pipeline.image_processor
            transformer = self.model.transformer
            vae = self.model.vae

            num_latent_channels = transformer.config.in_channels  # 16 for Anima
            latent_h = height // VAE_SCALE_FACTOR
            latent_w = width // VAE_SCALE_FACTOR
            cfg_enabled = cfg_scale > 1.0

            # ---- 1. text encoding (Qwen3 + AnimaTextConditioner) -----------
            self.model.text_encoder_to(self.train_device)
            self.model.text_conditioner_to(self.train_device)
            prompt_embeds, negative_prompt_embeds = self.__encode_prompts(
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg_enabled=cfg_enabled,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )
            self.model.text_encoder_to(self.temp_device)
            self.model.text_conditioner_to(self.temp_device)
            torch_gc()

            # ---- 2. latent noise -------------------------------------------
            # (B, C, T=1, H_lat, W_lat) -- the Cosmos transformer is 3D-shaped
            # but Anima uses T=1 for still images.
            latent_image = torch.randn(
                size=(1, num_latent_channels, 1, latent_h, latent_w),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )
            padding_mask = latent_image.new_zeros((1, 1, height, width), dtype=self.model.train_dtype.torch_dtype())

            # ---- 3. timesteps via flow-matching sigmas --------------------
            # Mirrors AnimaSetTimestepsStep in before_denoise.py: linear
            # sigmas from 1.0 to 1/N, passed to set_timesteps(sigmas=...).
            sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
            scheduler.set_timesteps(sigmas=sigmas, device=self.train_device)
            timesteps = scheduler.timesteps
            scheduler.set_begin_index(0)

            # ---- 4. denoise loop ------------------------------------------
            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                latent_model_input = latent_image.to(dtype=self.model.train_dtype.torch_dtype())
                # Normalized timestep per AnimaLoopBeforeDenoiser:
                #   t / scheduler.config.num_train_timesteps
                # broadcast to (B,) and cast to model dtype.
                t_norm = (timestep.expand(latent_image.shape[0]) / scheduler.config.num_train_timesteps).to(
                    self.model.train_dtype.torch_dtype()
                )

                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=t_norm,
                    encoder_hidden_states=prompt_embeds.to(self.model.train_dtype.torch_dtype()),
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

                if cfg_enabled:
                    noise_pred_uncond = transformer(
                        hidden_states=latent_model_input,
                        timestep=t_norm,
                        encoder_hidden_states=negative_prompt_embeds.to(self.model.train_dtype.torch_dtype()),
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

                latent_image = scheduler.step(
                    noise_pred,
                    timestep,
                    latent_image,
                    return_dict=False,
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # ---- 5. VAE decode --------------------------------------------
            self.model.vae_to(self.train_device)
            latents = self.model.unscale_latents(latent_image.to(vae.dtype))
            # vae.decode returns (B, C, T=1, H, W); strip the T axis for
            # the image_processor which wants (B, C, H, W).
            image = vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = image_processor.postprocess(image, output_type="pil")

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
            )

    def sample(
        self,
        sample_config: SampleConfig,
        destination: str,
        image_format: ImageFormat | None = None,
        video_format: VideoFormat | None = None,
        audio_format: AudioFormat | None = None,
        on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            # Cosmos requires (height, width) divisible by VAE_SCALE_FACTOR * 2
            # = 16 (per AnimaPrepareLatentsStep.check_inputs). 32 is a safer
            # rounding -- matches what other samplers use.
            height=self.quantize_resolution(sample_config.height, 32),
            width=self.quantize_resolution(sample_config.width, 32),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output,
            destination,
            image_format,
            video_format,
            audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, AnimaSampler, ModelType.ANIMA)
