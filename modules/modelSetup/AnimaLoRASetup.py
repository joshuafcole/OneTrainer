from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class AnimaLoRASetup(
    BaseAnimaSetup,
):
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

        # TI embeddings train alongside the LoRA. The high embedding_learning_rate
        # vs. the low-capacity adapter is what lets the token carry the conditional
        # binding while the adapter stays a gentle, always-on enrichment.
        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_embeddings(),
                    parameter_group_collection,
                    config.embedding_learning_rate,
                    "embeddings",
                )
            if config.text_encoder.train_embedding and model.text_conditioner is not None:
                # Fallback chain: explicit t5_embedding_learning_rate wins;
                # otherwise 0.1x the explicit Qwen LR; otherwise null, which
                # inherits the base learning_rate just like a null
                # embedding_learning_rate does for the Qwen group.
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

        # Only the Cosmos transformer is LoRA-tuned. AnimaTextConditioner
        # is held frozen -- it is the trained Qwen3->Cosmos adapter that
        # came with the original checkpoint, and the rmatif PR exposes
        # PeftAdapterMixin on it for users who want to fine-tune it
        # later, but we don't enable that yet.
        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer_lora, config.transformer
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

        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora, config.transformer, model.train_progress
        )

    def setup_model(
        self,
        model: AnimaModel,
        config: TrainConfig,
    ):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        # TI tokens attach to the Qwen3 word embedding table and the T5
        # input embedding table (model.text_conditioner.embed). Promote
        # both to the embedding dtype (fp32) so the frozen vocab slice
        # and the trainable vectors concatenate cleanly inside each
        # wrapper.
        if config.train_any_embedding():
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
        # Latent + text caches keep Qwen3 + VAE off the train device.
        # The conditioner runs every step and must be present. When
        # training a TI embedding, Qwen3 runs live every step (text cache
        # disabled), so it must stay resident even with latent_caching on.
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = config.train_text_encoder_or_embedding() or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.text_conditioner_to(self.train_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.text_conditioner.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
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


factory.register(BaseModelSetup, AnimaLoRASetup, ModelType.ANIMA, TrainingMethod.LORA)
