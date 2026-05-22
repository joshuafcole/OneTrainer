from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class AnimaEmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: AnimaModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
