from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.anima.AnimaEmbeddingLoader import AnimaEmbeddingLoader
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.util.enum.ModelType import ModelType

AnimaEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima-embedding.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=AnimaEmbeddingLoader,
)
