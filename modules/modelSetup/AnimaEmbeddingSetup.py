from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class AnimaEmbeddingSetup(
    BaseAnimaSetup,
):
    """Pure textual-inversion (embedding-only) training for Anima.

    Unlike AnimaLoRASetup -- which trains a LoRA adapter and lets a TI
    token ride alongside it -- this setup trains *only* the placeholder
    token's vector(s) in the Qwen3 word-embedding table. The Cosmos
    transformer, the AnimaTextConditioner, Qwen3, and the VAE are all
    frozen; the concept lives entirely in the trained token.

    Because the token is injected into Qwen3 (the T5 table lives inside
    the frozen conditioner and is left untouched), Qwen3 must run live
    every step so gradients reach the token vector -- the text cache is
    disabled for embedding runs (see config.train_text_encoder_or_embedding)
    and BaseAnimaSetup.predict passes tokens_qwen through the embedding
    wrapper instead of consuming a cached hidden state.
    """

    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train_embedding and model.text_encoder is not None:
            self._add_embedding_param_groups(
                model.all_text_encoder_embeddings(),
                parameter_group_collection,
                config.embedding_learning_rate,
                "embeddings",
            )

        if config.text_encoder.train_embedding and model.text_conditioner is not None:
            # Fallback chain: explicit t5_embedding_learning_rate wins; otherwise
            # 0.1x the explicit Qwen LR; otherwise null, which inherits the
            # base learning_rate just like a null embedding_learning_rate does
            # for the Qwen group.
            if config.t5_embedding_learning_rate is not None:
                t5_lr = config.t5_embedding_learning_rate
            elif config.embedding_learning_rate is not None:
                t5_lr = config.embedding_learning_rate * 0.1
            else:
                t5_lr = None
            self._add_embedding_param_groups(
                model.all_t5_embeddings(),
                parameter_group_collection,
                t5_lr,
                "t5-embeddings",
            )

        return parameter_group_collection

    def __setup_requires_grad(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder.requires_grad_(False)
        model.text_conditioner.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

    def setup_model(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        # TI tokens attach to the Qwen3 word embedding table and the T5
        # input embedding table (model.text_conditioner.embed). Promote
        # both to the embedding dtype (fp32) so the frozen vocab slice
        # and the trainable vectors concatenate cleanly inside each
        # wrapper. The T5 lookup site in AnimaTextConditioner already
        # casts the lookup output back to the source dtype, so the
        # promotion is safe.
        if model.text_encoder is not None:
            model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        if model.text_conditioner is not None:
            model.text_conditioner.embed.to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._remove_added_embeddings_from_tokenizer(model.t5_tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        # Qwen3 runs live every step (the text cache is disabled for
        # embedding training so gradients can reach the trainable token
        # vectors), so it stays resident even with latent_caching on.
        # The conditioner runs every step. The VAE is only needed on the
        # train device when latents aren't cached. The transformer is
        # frozen but still runs the forward pass, so it stays resident.
        vae_on_train_device = not config.latent_caching

        model.text_encoder_to(self.train_device)
        model.text_conditioner_to(self.train_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.text_conditioner.eval()
        model.vae.eval()
        model.transformer.eval()

    def after_optimizer_step(
        self,
        model: AnimaModel,
        config: TrainConfig,
        train_progress: TrainProgress,
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            if model.embedding_wrapper is not None:
                model.embedding_wrapper.normalize_embeddings()
            if model.t5_embedding_wrapper is not None:
                model.t5_embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, AnimaEmbeddingSetup, ModelType.ANIMA, TrainingMethod.EMBEDDING)
