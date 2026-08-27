from modules.util.enum.TrainingMethod import TrainingMethod


class EmbeddingName:
    def __init__(
            self,
            uuid: str,
            model_name: str,
    ):
        self.uuid = uuid
        self.model_name = model_name


class ModelNames:
    def __init__(
            self,
            base_model: str = "",
            prior_model: str = "",
            transformer_model: str = "",
            effnet_encoder_model: str = "",
            decoder_model: str = "",
            text_encoder_4: str = "",
            vae_model: str = "",
            lora: str = "",
            embedding: EmbeddingName | None = None,
            additional_embeddings: list[EmbeddingName] | None = None,
            include_text_encoder: bool = True,
            include_text_encoder_2: bool = True,
            include_text_encoder_3: bool = True,
            include_text_encoder_4: bool = True,
            include_unconditional_transformer: bool = True,
    ):
        self.base_model = base_model
        self.prior_model = prior_model
        self.transformer_model = transformer_model
        self.effnet_encoder_model = effnet_encoder_model
        self.decoder_model = decoder_model
        self.text_encoder_4 = text_encoder_4
        self.vae_model = vae_model
        self.lora = lora
        self.embedding = embedding
        self.additional_embeddings = [] if additional_embeddings is None else additional_embeddings
        self.include_text_encoder = include_text_encoder
        self.include_text_encoder_2 = include_text_encoder_2
        self.include_text_encoder_3 = include_text_encoder_3
        self.include_text_encoder_4 = include_text_encoder_4
        self.include_unconditional_transformer = include_unconditional_transformer

    def set_backup_path(self, training_method: TrainingMethod, backup_path: str) -> None:
        """Point the right name at a resumed backup directory.

        Which name that is follows what the run *saves*: a LoRA run backs up an
        adapter, an embedding run an embedding, everything else a full model.
        SLIDER saves as a LoRA -- same modules, same formats, same converters --
        so it resumes as one. Falling through to base_model would hand an adapter
        directory to the base-model loader: a resume that cannot work but reads
        like one that should, and is only discovered once the load fails.

        Lives here rather than at the two call sites (the trainer's resume and
        the sample window's) because they had drifted into two copies of the same
        three-way branch, and a new training method has to be right in both.
        """
        if training_method in (TrainingMethod.LORA, TrainingMethod.SLIDER):
            self.lora = backup_path
        elif training_method == TrainingMethod.EMBEDDING:
            self.embedding.model_name = backup_path
        else:  # fine-tunes
            self.base_model = backup_path

    def all_embedding(self):
        if self.embedding is not None:
            return self.additional_embeddings + [self.embedding]
        else:
            return self.additional_embeddings
