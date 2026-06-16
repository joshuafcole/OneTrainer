"""Datasetless data loader for prompt-pair Concept-Sliders training.

Prompt-pair sliders (CS Eq. 7/8) have no image dataset: the frozen base supplies
the guidance direction and ``AnimaSliderSetup.predict`` generates the noised
latent ``x_t`` synthetically. So this loader does not build an MGDS pipeline at
all -- it just emits ``slider_steps_per_epoch`` trivial batches that drive the
training loop's step count. Each batch carries only ``concept_type`` (all
STANDARD, so the trainer's prior-prediction/regularization paths stay inert);
the setup builds everything else from the slider config.

It implements the BaseDataLoader surface the trainer actually touches
(``get_data_set().start_next_epoch()`` / ``.approximate_length()`` and
``get_data_loader()`` as an iterable), bypassing BaseDataLoader.__init__'s
MGDS construction.

The (ANIMA, SLIDER) factory slot is owned by a small dispatcher at the bottom of
this module: the coordinate-labeled IMAGE regime trains on a real dataset, so it
routes to AnimaSliderImageDataLoader; the PROMPT_PAIR regime uses this
datasetless loader.
"""

from modules.dataLoader.AnimaSliderImageDataLoader import AnimaSliderImageDataLoader
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch


class _SliderDataSet:
    """Stand-in for the MGDS dataset. Only start_next_epoch/approximate_length
    are consulted by the trainer; both are cheap bookkeeping here."""

    def __init__(self, steps_per_epoch: int):
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.epoch = -1

    def start_next_epoch(self):
        self.epoch += 1

    def approximate_length(self) -> int:
        return self.steps_per_epoch


class _SliderDataLoaderIterable:
    """Yields the synthetic per-step batches. ``concept_type`` is sized to
    config.batch_size so the trainer's ``range(batch_size)`` comprehensions
    line up; the slider objective itself works one prompt triple per step."""

    def __init__(self, dataset: _SliderDataSet, batch_size: int):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))

    def __len__(self) -> int:
        return self.dataset.steps_per_epoch

    def __iter__(self):
        concept_type = [ConceptType.STANDARD.value] * self.batch_size
        for _ in range(self.dataset.steps_per_epoch):
            # A fresh dict each step; the setup ignores its contents beyond
            # concept_type, but the trainer reads batch.get("latent_image")
            # defensively (returns None -> no profiler token hint).
            yield {"concept_type": list(concept_type)}


class AnimaSliderDataLoader(BaseDataLoader):
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
        # Intentionally NOT calling super().__init__: BaseDataLoader builds an
        # MGDS dataset, which a datasetless slider has nothing to feed.
        self.train_device = train_device
        self.temp_device = temp_device

        steps = 1 if is_validation else config.slider_steps_per_epoch
        batch_size = 1 if is_validation else config.batch_size

        self.__ds = _SliderDataSet(steps)
        self.__dl = _SliderDataLoaderIterable(self.__ds, batch_size)

    def get_data_set(self) -> _SliderDataSet:
        return self.__ds

    def get_data_loader(self) -> _SliderDataLoaderIterable:
        return self.__dl

    def _create_dataset(self, config, model, model_setup, train_progress, is_validation):
        # Abstract in BaseDataLoader; unused here because __init__ is overridden.
        return None


def _create_slider_data_loader(
        train_device: torch.device,
        temp_device: torch.device,
        config: TrainConfig,
        model,
        model_setup,
        train_progress: TrainProgress,
        is_validation: bool = False,
):
    """Dispatch the (ANIMA, SLIDER) loader by regime: coordinate-labeled IMAGE
    sliders train on a real dataset (AnimaSliderImageDataLoader); prompt-pair
    sliders use this datasetless step-driver."""
    if config.slider_regime == SliderRegime.IMAGE:
        return AnimaSliderImageDataLoader(
            train_device, temp_device, config, model, model_setup, train_progress, is_validation,
        )
    return AnimaSliderDataLoader(
        train_device, temp_device, config, model, model_setup, train_progress, is_validation,
    )


factory.register(BaseDataLoader, _create_slider_data_loader, ModelType.ANIMA, TrainingMethod.SLIDER)
