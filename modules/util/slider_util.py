"""Registration helpers shared by the Concept-Sliders hosts."""

from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util import factory
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def alias_lora_persistence_to_slider(*model_types: ModelType) -> None:
    """Point (model_type, SLIDER) at the model's existing LoRA saver and loader.

    A slider IS a LoRA on disk -- same modules, same formats, same converters --
    only the training objective differs. create_model_saver / create_model_loader
    do not fall back to a model-type-only entry (unlike the sampler and the data
    loader), so SLIDER needs its own registration or a finished run cannot save.

    Import order makes this safe: create.py imports modelSaver and modelLoader
    before modelSetup, so the LoRA entries exist by the time a slider setup module
    is imported. If one is missing that is a wiring bug, and raising here is much
    cheaper than discovering it after a run finishes training.
    """
    for model_type in model_types:
        for base_cls in (BaseModelSaver, BaseModelLoader):
            lora_impl = factory.get(base_cls, model_type, TrainingMethod.LORA)
            if lora_impl is None:
                raise RuntimeError(
                    f"cannot register a slider for {model_type}: no {base_cls.__name__} is "
                    f"registered for (in {model_type}, {TrainingMethod.LORA}). A slider saves "
                    f"as a LoRA, so the LoRA entry must exist first."
                )
            factory.register(base_cls, lora_impl, model_type, TrainingMethod.SLIDER)
