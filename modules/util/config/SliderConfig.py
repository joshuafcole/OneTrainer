import uuid
from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class SliderPromptConfig(BaseConfig):
    """One slider prompt triple for prompt-pair Concept Sliders.

    The frozen base supplies the guidance direction ``v(c+) - v(c-)`` at the
    neutral target concept ``c_t`` (``target``). With a non-empty preservation
    set (TrainConfig.slider_preservation_prompts) the direction is averaged over
    the preservation-augmented pairs (CS Eq. 8); with an empty set it reduces to
    the bare ``positive``/``negative`` pair (CS Eq. 7).

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
    """One declared slider axis for coordinate-labeled image sliders (docs §10).

    The training data is vanilla OneTrainer concepts whose captions carry an
    a1111-style coordinate token per image, e.g. ``(distance:-2)``. ``name`` is
    the token key matched (case-insensitively) and stripped from the caption
    before tokenization, so the conditioning stays orthogonal to the axis.

    Exactly one enabled axis must be flagged ``is_target``: its per-image
    coordinate ``value`` becomes the training-time adapter multiplier
    ``m = gain_k * value`` (coordinate-scaled reconstruction). The remaining
    declared axes are still stripped from the caption (so confounders the user
    knows about are kept out of the conditioning) and, when ``stratify`` is set,
    are reserved for the balanced sampler (a fast-follow; unused in v1).

    ``gain_k`` is the global gain mapping raw coordinate units onto the adapter
    multiplier; v1 consumes coordinates as-is (ordinal recommended but not
    enforced), so a per-axis rescale lives in dataset prep.
    """

    uuid: str
    enabled: bool
    name: str
    gain_k: float
    is_target: bool
    stratify: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("enabled", True, bool, False))
        data.append(("name", "distance", str, False))
        data.append(("gain_k", 1.0, float, False))
        data.append(("is_target", True, bool, False))
        data.append(("stratify", False, bool, False))

        return SliderAxisConfig(data)
