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


@factory.register(BaseModelSetup, ModelType.ANIMA, TrainingMethod.EMBEDDING)
class AnimaEmbeddingSetup(
    BaseAnimaSetup,
):
    # Pure textual inversion: the transformer, AnimaTextConditioner, Qwen3 and the VAE are all frozen, so
    # the concept lives entirely in the placeholder's rows of the Qwen3 word-embedding table. Qwen3 still
    # runs live every step -- the text cache is off whenever an embedding is trained (see
    # config.train_text_encoder_or_embedding) -- because that forward is the only path gradient can take
    # back to the token.

    def create_parameters(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._reject_output_embeddings(config)

        if config.text_encoder.train_embedding:
            self._add_embedding_param_groups(
                model.all_text_encoder_embeddings(), parameter_group_collection,
                config.embedding_learning_rate, "embeddings",
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
        model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

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
        vae_on_train_device = not config.latent_caching

        # The transformer is frozen but still runs the forward pass the loss is taken from, so it stays
        # resident. So does the text encoder, unconditionally: an EMBEDDING run turns the text cache off
        # whatever text_encoder.train_embedding says (train_text_encoder_or_embedding() is true for any
        # single-text-encoder model as soon as an embedding is being trained), so Qwen3 has to encode live
        # every step. It also carries AnimaTextConditioner with it -- the conditioner sits between Qwen3
        # and the transformer and runs on every step of every configuration, and evicting the text encoder
        # evicts it too.
        parts = ["transformer", "text_encoder"]
        if vae_on_train_device:
            parts.append("vae")
        model.materialize_only(*parts)

        model.text_encoder.eval()
        model.text_conditioner.eval()
        model.vae.eval()
        model.transformer.eval()

    def after_optimizer_step(
            self,
            model: AnimaModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            if model.embedding_wrapper is not None:
                model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
