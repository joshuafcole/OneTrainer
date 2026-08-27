"""Datasetless data loader for prompt-pair Concept-Sliders training.

Prompt-pair sliders (CS Eq. 7/8) have no image dataset: the frozen base supplies
the guidance direction and the host setup generates the noised latent x_t
synthetically. So this loader builds no MGDS pipeline at all -- it emits
``slider_steps_per_epoch`` trivial batches whose only job is to drive the
training loop's step count.

Each batch carries only ``concept_type``, all ``STANDARD``. That keeps the
trainer's prior-prediction and counterexample-repulsion paths inert: both select
rows by concept type, find none, and skip the extra frozen forward they would
otherwise run.

It implements the BaseDataLoader surface the trainer actually touches
(``get_data_set().start_next_epoch()`` / ``.approximate_length()`` and
``get_data_loader()`` as an iterable), bypassing BaseDataLoader.__init__'s MGDS
construction.

The loader is model-agnostic -- it emits no model-specific tensors -- so the same
class serves every model type with a slider setup. It does not register itself:
SliderDataLoader owns the (model type, SLIDER) slot and dispatches to this class
for the PROMPT_PAIR regime.
"""

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

import torch


class _SliderDataSet:
    """Stand-in for the MGDS dataset. The trainer consults only
    start_next_epoch/approximate_length; both are cheap bookkeeping here."""

    def __init__(self, steps_per_epoch: int):
        self.steps_per_epoch = max(0, int(steps_per_epoch))
        self.epoch = -1

    def start_next_epoch(self):
        self.epoch += 1

    def approximate_length(self) -> int:
        return self.steps_per_epoch


class _SliderDataLoaderIterable:
    """Yields the synthetic per-step batches. ``concept_type`` is sized to
    config.batch_size so the trainer's ``range(batch_size)`` comprehensions line
    up; the slider objective itself works one prompt triple per step."""

    def __init__(self, dataset: _SliderDataSet, batch_size: int):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))

    def __len__(self) -> int:
        return self.dataset.steps_per_epoch

    def __iter__(self):
        concept_type = [ConceptType.STANDARD.value] * self.batch_size
        for _ in range(self.dataset.steps_per_epoch):
            yield {"concept_type": list(concept_type)}


class SliderPromptPairDataLoader(BaseDataLoader):
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
        # Intentionally NOT calling super().__init__: BaseDataLoader builds an MGDS
        # dataset, which a datasetless slider has nothing to feed.
        self.train_device = train_device
        self.temp_device = temp_device

        # A prompt-pair slider has no held-out data. An epoch length of 0 makes the
        # trainer's validation pass return before it touches a batch -- which it
        # must, because it reads concept_name / concept_path / concept_seed off one,
        # and because a "validation loss" resampled from the same synthetic
        # distribution as training would be noise reported as a metric.
        steps = 0 if is_validation else config.slider_steps_per_epoch

        self.__ds = _SliderDataSet(steps)
        self.__dl = _SliderDataLoaderIterable(self.__ds, config.batch_size)

    def get_data_set(self) -> _SliderDataSet:
        return self.__ds

    def get_data_loader(self) -> _SliderDataLoaderIterable:
        return self.__dl

    def _create_dataset(self, config, model, model_setup, train_progress, is_validation):
        # Abstract in BaseDataLoader; unused here because __init__ is overridden.
        return None
