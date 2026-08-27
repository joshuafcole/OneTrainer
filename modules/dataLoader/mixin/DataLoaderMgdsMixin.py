import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


def dataset_concepts(config: TrainConfig, is_validation: bool) -> list[ConceptConfig]:
    """The concepts that actually feed one dataset.

    config.concepts is populated only by callers that build a TrainConfig in
    memory; in the normal UI and CLI paths it is None and the concepts live in the
    file config.concept_file_name names. Anything that wants to reason about the
    dataset -- not only the pipeline itself -- has to resolve it the same way, so
    the resolution lives here rather than inline in _create_mgds.
    """
    concepts = config.concepts
    if concepts is None:
        with open(config.concept_file_name, 'r') as f:
            concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

    # choose all validation concepts, or none of them, depending on is_validation
    return [concept for concept in concepts if (ConceptType(concept.type) == ConceptType.VALIDATION) == is_validation]


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        # convert before passing to MGDS
        concepts = [c.to_dict() for c in dataset_concepts(config, is_validation)]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        # Just defaults for now.
        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
