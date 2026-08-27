"""The (model type, SLIDER) data-loader slot, dispatched by slider regime.

The two slider regimes want opposite things from a data loader. PROMPT_PAIR has
no dataset at all -- SliderPromptPairDataLoader emits bare step-driver batches.
IMAGE trains on an ordinary image dataset and wants the model's full MGDS
pipeline, plus the caption coordinate threaded through it.

The factory keys on (model type, training method) and knows nothing about a
regime, so one entry has to serve both and pick at construction. This class is
that entry: it builds the right loader and forwards the two methods the trainer
actually uses.
"""

from modules.dataLoader.AnimaSliderImageDataLoader import AnimaSliderImageDataLoader
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.SliderPromptPairDataLoader import SliderPromptPairDataLoader
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

# Model types with a slider setup. Without an entry here a slider run falls back
# to the model-type-only loader -- which, for PROMPT_PAIR, means an MGDS pipeline
# over a dataset that does not exist.
SLIDER_MODEL_TYPES = (
    ModelType.ANIMA,
    ModelType.STABLE_DIFFUSION_XL_10_BASE,
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)

# Per-model IMAGE-regime loaders. A model absent from this map has a prompt-pair
# slider host and no coordinate-labeled one, which is a real gap and not a
# fallback: the ordinary loader would emit no slider_coordinate and the setup
# would fail on a missing batch key several minutes into the run.
IMAGE_REGIME_LOADERS = {
    ModelType.ANIMA: AnimaSliderImageDataLoader,
}


class SliderDataLoader(BaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model,
            model_setup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        # Intentionally NOT calling super().__init__: it would build a second,
        # unused MGDS dataset alongside the delegate's.
        self.train_device = train_device
        self.temp_device = temp_device
        self._impl = self._create_impl(config)(
            train_device, temp_device, config, model, model_setup, train_progress, is_validation,
        )

    @staticmethod
    def _create_impl(config: TrainConfig) -> type[BaseDataLoader]:
        if config.slider_regime != SliderRegime.IMAGE:
            return SliderPromptPairDataLoader

        impl = IMAGE_REGIME_LOADERS.get(config.model_type)
        if impl is None:
            raise NotImplementedError(
                f"the coordinate-labeled image slider regime is not implemented for "
                f"{config.model_type}. Pick the prompt-pair regime on the Slider tab, or train "
                f"the image regime on a model that supports it "
                f"({', '.join(str(t) for t in IMAGE_REGIME_LOADERS)})."
            )
        return impl

    def get_data_set(self):
        return self._impl.get_data_set()

    def get_data_loader(self):
        return self._impl.get_data_loader()

    def _create_dataset(self, config, model, model_setup, train_progress, is_validation):
        # Abstract in BaseDataLoader; unused here because __init__ is overridden.
        return None


for _model_type in SLIDER_MODEL_TYPES:
    factory.register(BaseDataLoader, SliderDataLoader, _model_type, TrainingMethod.SLIDER)
