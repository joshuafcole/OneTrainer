import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.AnimaModel import PROMPT_MAX_LENGTH, AnimaModel
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.bucket_limits import ANIMA_MAX_BUCKET_RESOLUTION
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.thread_safety import apply_thread_safe_forward
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.Tokenize import Tokenize


class AnimaBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    """Anima MGDS dataloader.

    Caches what is expensive (Qwen3 last_hidden_state, ~1 MB per
    sample at 512 tokens) and leaves the cheap AnimaTextConditioner
    pass to training step time. This trade exists because:

      - Qwen3-0.6B is a 1.2 GB model -- not free to run per step.
      - AnimaTextConditioner is 269 MB and consumes already-cached
        Qwen3 hidden states + T5 token ids; running it per step is
        ~10 ms and avoids caching its (deterministic) output, which
        would double text-cache footprint.

    Cached tensors per sample (typical 512-token prompt, bf16):

      tokens_qwen / tokens_mask_qwen :        512 int64  (8 KB)
      text_encoder_hidden_state      :  512 x 1024 bf16  (~1 MB)
      tokens_t5 / tokens_mask_t5     :        512 int64  (8 KB)
      latent_image                   :   16 x 1 x H x W  (model-dependent)

    Note that mgds EncodeQwenText was written for Qwen2.5-VL /
    Qwen3ForCausalLM and does not enforce the type annotation;
    duck-typing it against base Qwen3Model works because the call
    contract (input_ids, attention_mask, output_hidden_states=True)
    is the same.
    """

    def _preparation_modules(self, config: TrainConfig, model: AnimaModel):
        rescale_image = RescaleImageChannels(
            image_in_name="image",
            image_out_name="image",
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        encode_image = EncodeVAE(
            in_name="image",
            out_name="latent_image_distribution",
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        image_sample = SampleVAEDistribution(
            in_name="latent_image_distribution",
            out_name="latent_image",
            mode="mean",
        )

        # Substitute TI placeholder strings with their per-encoder
        # generated tokens before tokenizing. Each encoder gets its own
        # substitution (mirrors Flux): Qwen3 ids only live in
        # model.tokenizer, T5 ids only live in model.t5_tokenizer.
        add_embeddings_to_prompt_qwen = MapData(
            in_name="prompt",
            out_name="prompt_qwen",
            map_fn=model.add_text_encoder_embeddings_to_prompt,
        )
        add_embeddings_to_prompt_t5 = MapData(
            in_name="prompt",
            out_name="prompt_t5",
            map_fn=model.add_t5_embeddings_to_prompt,
        )

        # Two tokenizers, two per-encoder substituted prompts: matches
        # the upstream AnimaTextEncoderStep which tokenizes the prompt
        # twice (Qwen2 BPE vs. T5 SentencePiece) and feeds both into the
        # conditioner.  No chat template -- Anima trained on plain
        # prompts, unlike Z-Image.
        max_token_length = config.text_encoder_sequence_length or PROMPT_MAX_LENGTH
        tokenize_qwen = Tokenize(
            in_name="prompt_qwen",
            tokens_out_name="tokens_qwen",
            mask_out_name="tokens_mask_qwen",
            tokenizer=model.tokenizer,
            max_token_length=max_token_length,
        )
        tokenize_t5 = Tokenize(
            in_name="prompt_t5",
            tokens_out_name="tokens_t5",
            mask_out_name="tokens_mask_t5",
            tokenizer=model.t5_tokenizer,
            max_token_length=max_token_length,
        )

        if config.dataloader_threads > 1:
            apply_thread_safe_forward(model.text_encoder)  # workaround for transformers#42673

        # hidden_state_output_index=-1: matches encoders.py's use of
        # `last_hidden_state`, which equals `hidden_states[-1]` for
        # decoder-only LMs like Qwen3 (final norm is already applied).
        encode_qwen = EncodeQwenText(
            tokens_name="tokens_qwen",
            tokens_attention_mask_in_name="tokens_mask_qwen",
            hidden_state_out_name="text_encoder_hidden_state",
            tokens_attention_mask_out_name="tokens_mask_qwen",
            text_encoder=model.text_encoder,
            hidden_state_output_index=-1,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )

        modules = [
            rescale_image,
            encode_image,
            image_sample,
            add_embeddings_to_prompt_qwen,
            add_embeddings_to_prompt_t5,
            tokenize_qwen,
            tokenize_t5,
        ]
        # When training a TI embedding, Qwen3 must run live at step time
        # (under grad), so skip pre-computing/caching its hidden states.
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_qwen)
        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        image_split_names = ["latent_image", "original_resolution", "crop_offset"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        # When training a TI embedding, nothing text-side is cached: tokens
        # are re-derived live each step and Qwen3 runs under grad so the
        # trainable token vectors receive gradients.
        text_split_names = []
        if not config.train_text_encoder_or_embedding():
            text_split_names = [
                "tokens_qwen",
                "tokens_mask_qwen",
                "tokens_t5",
                "tokens_mask_t5",
                "text_encoder_hidden_state",
            ]
        sort_names = (
            image_aggregate_names
            + image_split_names
            + [
                "prompt",
                "tokens_qwen",
                "tokens_mask_qwen",
                "tokens_t5",
                "tokens_mask_t5",
                "text_encoder_hidden_state",
                "concept",
            ]
        )

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        output_names = [
            "image_path",
            "latent_image",
            "prompt",
            "tokens_qwen",
            "tokens_mask_qwen",
            "tokens_t5",
            "tokens_mask_t5",
            "original_resolution",
            "crop_resolution",
            "crop_offset",
        ]
        # text_encoder_hidden_state only exists when not training an
        # embedding (otherwise Qwen3 runs live in predict()).
        if not config.train_text_encoder_or_embedding():
            output_names.append("text_encoder_hidden_state")

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: AnimaModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(
            in_name="latent_image",
            out_name="decoded_image",
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        decode_prompt = DecodeTokens(
            in_name="tokens_qwen",
            out_name="decoded_prompt",
            tokenizer=model.tokenizer,
        )
        save_image = SaveImage(
            image_in_name="decoded_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
            before_save_fun=before_save_fun,
        )
        save_prompt = SaveText(
            text_in_name="decoded_prompt",
            original_path_in_name="image_path",
            path=debug_dir,
            before_save_fun=before_save_fun,
        )
        return [decode_image, save_image, decode_prompt, save_prompt]

    def _create_dataset(
        self,
        config: TrainConfig,
        model: BaseModel,
        model_setup: BaseModelSetup,
        train_progress: TrainProgress,
        is_validation: bool = False,
    ):
        # vae_frame_dim=True inserts an ImageToVideo module before the
        # VAE encode, which adds a T=1 axis so AutoencoderKLQwenImage's
        # 3D-shaped encoder (which expects (B, C, T, H, W)) gets a 5D
        # input it can patch through. HunyuanVideo does the same for
        # the same reason.
        #
        # aspect_bucketing_quantization=64: Cosmos requires (H, W)
        # divisible by vae_scale_factor*2 = 16; we pick 64 to align
        # with ZImage / Flux conventions and keep aspect buckets sane.
        #
        # aspect_bucketing_max_resolution caps the bucket long edge to keep
        # extreme aspect rungs within Cosmos's RoPE range; see the constant.
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            aspect_bucketing_quantization=64,
            vae_frame_dim=True,
            aspect_bucketing_max_resolution=ANIMA_MAX_BUCKET_RESOLUTION,
        )


factory.register(BaseDataLoader, AnimaBaseDataLoader, ModelType.ANIMA)
