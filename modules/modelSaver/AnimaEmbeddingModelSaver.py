from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.anima.AnimaEmbeddingSaver import AnimaEmbeddingSaver
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.util.enum.ModelType import ModelType

AnimaEmbeddingModelSaver = make_embedding_model_saver(
    ModelType.ANIMA,
    model_class=AnimaModel,
    embedding_saver_class=AnimaEmbeddingSaver,
)
