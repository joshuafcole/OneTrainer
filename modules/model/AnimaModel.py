import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.modular_pipelines.anima import AnimaAutoBlocks
from transformers import Qwen2TokenizerFast, Qwen3Model, T5TokenizerFast

# Anima caps prompt length at 4096; the AnimaTextEncoderStep default and
# the value the upstream pipeline ships with is 512. Match it so cached
# embeddings line up with what sampling will produce.
PROMPT_MAX_LENGTH = 512


class AnimaModelEmbedding:
    def __init__(
        self,
        uuid: str,
        text_encoder_vector: Tensor | None,
        t5_vector: Tensor | None,
        placeholder: str,
        is_output_embedding: bool,
    ):
        # Anima injects textual-inversion tokens into both the Qwen3 word
        # embedding table (model.text_encoder.get_input_embeddings()) and
        # the T5 input embedding table (model.text_conditioner.embed).
        # Each encoder gets its own trainable vector with its own
        # placeholder string -- the substitution path is per-encoder so
        # the two tokenizers can register independent token ids.
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )
        self.t5_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=t5_vector,
            is_output_embedding=False,
        )


class AnimaModel(BaseModel):
    """Container for the components of a converted Anima checkpoint.

    Anima is a text-to-image model built on the Cosmos Predict2 DiT
    architecture. Its inference graph is the only standout vs. other
    diffusers-style models:

      prompt -> Qwen2 tokenizer  -> Qwen3Model.last_hidden_state -+
                                                                   |
                                                                   +--> AnimaTextConditioner --> encoder_hidden_states
                                                                   |
      prompt -> T5  tokenizer    -> t5_input_ids ------------------+

      latents (B,16,1,H,W) + encoder_hidden_states ---> CosmosTransformer3DModel ---> noise_pred
      latents ---> AutoencoderKLQwenImage.decode ---> image

    Note that there is no T5 encoder model to load -- the T5 input_ids
    are looked up in an embedding table that lives inside
    AnimaTextConditioner (model.text_conditioner.embed). Textual
    inversion injects trainable rows into both the Qwen3 word embedding
    table and that T5 embedding table.
    """

    # base model data
    tokenizer: Qwen2TokenizerFast | None
    t5_tokenizer: T5TokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3Model | None
    text_conditioner: AnimaTextConditioner | None
    vae: AutoencoderKLQwenImage | None
    transformer: CosmosTransformer3DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data. TI tokens attach to both the
    # Qwen3 word embedding table (model.text_encoder.get_input_embeddings())
    # and the T5 input embedding table (model.text_conditioner.embed),
    # with separate trainable vectors and a separate wrapper per encoder.
    embedding: AnimaModelEmbedding | None
    additional_embeddings: list[AnimaModelEmbedding] | None
    embedding_wrapper: AdditionalEmbeddingWrapper | None
    t5_embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data. Only the Cosmos transformer is
    # LoRA-targeted; AnimaTextConditioner is frozen (it's a learned
    # adapter from Qwen3 hidden states to Cosmos cross-attention input
    # and its weights came from net.llm_adapter.* in the original
    # checkpoint), and so is Qwen3.
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
        self,
        model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.t5_tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.text_conditioner = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32  # TODO

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper = None
        self.t5_embedding_wrapper = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [
            a
            for a in [
                self.transformer_lora,
            ]
            if a is not None
        ]

    def all_embeddings(self) -> list[AnimaModelEmbedding]:
        return self.additional_embeddings + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] + (
            [self.embedding.text_encoder_embedding] if self.embedding is not None else []
        )

    def all_t5_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.t5_embedding for embedding in self.additional_embeddings] + (
            [self.embedding.t5_embedding] if self.embedding is not None else []
        )

    def add_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_embeddings(), prompt)

    def add_t5_embeddings_to_prompt(self, prompt: str) -> str:
        # Only substitute embeddings that actually have a T5-side vector.
        # When T5-side training is off (the default), the t5_embedding has
        # no vector and an empty joint_text_tokens; substituting it would
        # strip the placeholder string from the prompt. Skipping it leaves
        # the placeholder to tokenize naturally on the T5 side -- the same
        # thing ComfyUI's Anima encoder does, since it never injects a
        # T5-side TI vector.
        active = [embedding for embedding in self.all_t5_embeddings() if embedding.vector is not None]
        return self._add_embeddings_to_prompt(active, prompt)

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        # Move ONLY Qwen3. The text_conditioner has its own to(); training
        # needs the conditioner on the train device every step (cached
        # qwen_hidden_states -> conditioner -> transformer), while Qwen3
        # itself stays on the temp device when latent_caching is on.
        if self.text_encoder is not None:
            if (
                self.text_encoder_offload_conductor is not None
                and self.text_encoder_offload_conductor.layer_offload_activated()
            ):
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def text_conditioner_to(self, device: torch.device):
        if self.text_conditioner is not None:
            self.text_conditioner.to(device=device)

    def transformer_to(self, device: torch.device):
        if (
            self.transformer_offload_conductor is not None
            and self.transformer_offload_conductor.layer_offload_activated()
        ):
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.text_conditioner_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.text_conditioner is not None:
            self.text_conditioner.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        """Return an upstream AnimaModularPipeline holding our components.

        Anima only ships a modular pipeline (no traditional
        AnimaPipeline class). We wire our pre-loaded components into a
        fresh AnimaAutoBlocks-based pipeline; this gives samplers and
        debug scripts the standard pipe(prompt=...) entry point.
        """
        pipe = AnimaAutoBlocks().init_pipeline()
        pipe.update_components(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            text_conditioner=self.text_conditioner,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.noise_scheduler,
        )
        return pipe

    def encode_text(
        self,
        train_device: torch.device,
        batch_size: int = 1,
        rand: Random | None = None,
        text: str | list[str] = None,
        tokens_qwen: Tensor = None,
        tokens_mask_qwen: Tensor = None,
        tokens_t5: Tensor = None,
        tokens_mask_t5: Tensor = None,
        text_encoder_dropout_probability: float | None = None,
        qwen_hidden_states: Tensor = None,
        text_encoder_sequence_length: int | None = None,
    ) -> Tensor:
        """Produce the Cosmos transformer's encoder_hidden_states from a prompt.

        Mirrors the chain in
        ``diffusers.modular_pipelines.anima.encoders.AnimaTextEncoderStep`` and
        ``before_denoise.AnimaTextConditioningStep`` so that cached training
        embeddings match the upstream sampler's outputs exactly:

          1. Tokenize the prompt with Qwen2TokenizerFast (BPE).
          2. Tokenize the same prompt with T5TokenizerFast (SentencePiece).
          3. Run Qwen3Model and take ``last_hidden_state``.
          4. Zero out padding positions on the Qwen3 output (matches
             upstream's ``prompt_embeds * mask.unsqueeze(-1)``).
          5. Feed (qwen_hidden_states, t5_input_ids, both masks) into
             AnimaTextConditioner to get the encoder_hidden_states the
             Cosmos transformer's cross-attention consumes.

        ``tokens_*`` / ``qwen_hidden_states`` are intermediate-cache
        entry points -- if any are passed they short-circuit earlier
        stages, matching the conventions used by Flux's data loader.
        Caching strategy across stages:

          - Stage A (cheapest, biggest cache): ``tokens_*`` only --
            saves tokenization, runs Qwen3 + text_conditioner each step.
          - Stage B (recommended): ``qwen_hidden_states + tokens_t5*`` --
            skips Qwen3 (1.2 GB encoder), runs only the 269 MB
            conditioner each step.
          - Stage C: nothing cached; pure on-the-fly.
        """
        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            # Modular pipeline implements unconditional generation by
            # encoding a second "negative_prompt" prompt, not by dropping
            # an existing one. Honoring text-encoder dropout would mean
            # replacing the per-sample prompt with "" at random, which
            # is doable but not what other OneTrainer models do.
            raise NotImplementedError("text encoder dropout is not supported for Anima yet")

        # Replace textual-inversion placeholder strings with their per-
        # encoder generated tokens before tokenizing. Each encoder gets
        # its own substitution (mirrors Flux's two-encoder TI): the Qwen3
        # placeholder strings only have ids in model.tokenizer, the T5
        # placeholder strings only have ids in model.t5_tokenizer. The
        # same substitution runs at sample time so train/inference stay
        # aligned.
        text_qwen = None
        text_t5 = None
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text_qwen = [self.add_text_encoder_embeddings_to_prompt(t) for t in text]
            text_t5 = [self.add_t5_embeddings_to_prompt(t) for t in text]

        max_length = text_encoder_sequence_length if text_encoder_sequence_length else PROMPT_MAX_LENGTH

        # ---- stage A: tokenize -------------------------------------------------
        if tokens_qwen is None and text_qwen is not None:
            qwen_inputs = self.tokenizer(
                text_qwen,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens_qwen = qwen_inputs.input_ids.to(self.text_encoder.device)
            tokens_mask_qwen = qwen_inputs.attention_mask.to(self.text_encoder.device)

        if tokens_t5 is None and text_t5 is not None:
            t5_inputs = self.t5_tokenizer(
                text_t5,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens_t5 = t5_inputs.input_ids.to(self.text_conditioner.device)
            tokens_mask_t5 = t5_inputs.attention_mask.to(self.text_conditioner.device)

        # ---- stage B: Qwen3 hidden states --------------------------------------
        if qwen_hidden_states is None:
            with self.text_encoder_autocast_context:
                qwen_out = self.text_encoder(
                    input_ids=tokens_qwen,
                    attention_mask=tokens_mask_qwen,
                    output_hidden_states=False,
                    use_cache=False,
                )
                qwen_hidden_states = qwen_out.last_hidden_state

        # Always mask -- cached qwen_hidden_states may be stored unmasked
        # (the mgds EncodeQwenText module returns the raw layer output).
        # Matches encoders.py:135.
        qwen_hidden_states = qwen_hidden_states * tokens_mask_qwen.to(qwen_hidden_states).unsqueeze(-1)

        # ---- stage C: AnimaTextConditioner -------------------------------------
        # The conditioner is a frozen, trained adapter. Move inputs onto
        # its device + dtype so users can keep Qwen3 / T5 / conditioner
        # on different devices without explicit shuffling.
        cond_device = self.text_conditioner.device
        cond_dtype = self.text_conditioner.dtype
        encoder_hidden_states = self.text_conditioner(
            source_hidden_states=qwen_hidden_states.to(device=cond_device, dtype=cond_dtype),
            source_attention_mask=tokens_mask_qwen.to(cond_device),
            target_input_ids=tokens_t5.to(cond_device),
            target_attention_mask=tokens_mask_t5.to(cond_device),
        )
        return encoder_hidden_states.to(dtype=self.transformer.dtype, device=train_device)

    # ----- VAE latent space (-) ------------------------------------------------
    # AutoencoderKLQwenImage stores per-channel normalization in its config
    # (latents_mean[16], latents_std[16]). Anima's decoders.py uses
    #
    #     latents_std = 1.0 / std
    #     latents = latents / latents_std + latents_mean    # = latents * std + mean
    #
    # which is the canonical "un-normalize" direction. The training-time
    # direction (image latents -> network input) is the inverse.

    def _latents_mean_std(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        # (1, z_dim, 1, 1, 1) so we broadcast against (B, z_dim, T, H, W).
        z_dim = self.vae.config.z_dim
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, z_dim, 1, 1, 1
        )
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, z_dim, 1, 1, 1
        )
        return mean, std

    def scale_latents(self, latents: Tensor) -> Tensor:
        """VAE-output latents -> training/inference network input space."""
        mean, std = self._latents_mean_std(latents)
        return (latents - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        """Network-output latents -> VAE decoder input space."""
        mean, std = self._latents_mean_std(latents)
        return latents * std + mean

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        # Cosmos Predict2's FlowMatchEulerDiscreteScheduler ships with
        # base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5,
        # max_shift=1.15 and time_shift_type="exponential" -- the same
        # linear-in-mu, exp-out formula as Flux/SD3. shift values at the
        # common training resolutions: ~1.88 (512), ~3.16 (1024), ~7.51 (1536).
        assert self.noise_scheduler is not None and self.transformer is not None
        cfg = self.noise_scheduler.config
        base_seq = cfg["base_image_seq_len"]
        max_seq = cfg["max_image_seq_len"]
        base_shift = cfg["base_shift"]
        max_shift = cfg["max_shift"]
        patch_size = self.transformer.config["patch_size"]
        patch_h, patch_w = patch_size[-2], patch_size[-1]

        image_seq_len = (latent_width // patch_w) * (latent_height // patch_h)
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        mu = image_seq_len * m + b
        return math.exp(mu)
