import uuid
from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class SliderPromptConfig(BaseConfig):
    """One slider prompt triple for prompt-pair Concept Sliders.

    The frozen base supplies the guidance direction ``v(c+) - v(c-)`` at the
    neutral target concept ``c_t`` (``target``). With a non-empty preservation
    set (``TrainConfig.slider_preservation_prompts``) the direction is averaged
    over the preservation-augmented pairs (CS Eq. 8); with an empty set it
    reduces to the bare ``positive``/``negative`` pair (CS Eq. 7).

    ``weight`` lets several triples be mixed with unequal sampling probability
    when more than one attribute axis is trained into the same slider.
    """

    uuid: str
    enabled: bool
    target: str
    positive: str
    negative: str
    weight: float

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("enabled", True, bool, False))
        data.append(("target", "a portrait photo of a person", str, False))
        data.append(("positive", "a portrait photo of an old person", str, False))
        data.append(("negative", "a portrait photo of a young person", str, False))
        data.append(("weight", 1.0, float, False))

        return SliderPromptConfig(data)


class SliderAxisConfig(BaseConfig):
    """One declared axis for the coordinate-labeled image slider regime.

    The dataset is ordinary OneTrainer concepts whose captions carry an
    a1111-style coordinate token per image, e.g. ``(distance:-2)``. ``name`` is
    the token key: it is matched case-insensitively and stripped out of the
    caption before tokenization, so the conditioning stays orthogonal to the
    axis.

    Exactly one enabled axis carries ``is_target``. Its per-image coordinate
    ``value`` becomes the training-time adapter multiplier
    ``m = gain_k * value``. Any other declared axis is still stripped from the
    caption -- that is what declaring it is *for*: a confounder the user has
    labelled stays out of the conditioning even though this run is not training
    it.

    ``gain_k`` maps raw coordinate units onto that multiplier. Coordinates are
    consumed as authored, so the dataset can be labelled in whatever units it was
    measured in and the gain is what brings the extremes to roughly +/-1 -- and
    it can be retuned without re-captioning or rebuilding the latent cache.
    """

    uuid: str
    enabled: bool
    name: str
    gain_k: float
    is_target: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("enabled", True, bool, False))
        data.append(("name", "", str, False))
        data.append(("gain_k", 1.0, float, False))
        data.append(("is_target", True, bool, False))

        return SliderAxisConfig(data)
