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


class SliderImagePairConfig(BaseConfig):
    """One before/after image pair for image-pair (visual) Concept Sliders.

    No text guidance: the flow-matching reconstruction target itself is the
    supervision (CS Eq. 9). The negative-scaled adapter is trained to
    reconstruct ``before`` (the A pole) and the positive-scaled adapter to
    reconstruct ``after`` (the B pole), both under the same conditioning, so the
    slider's +/- directions align with the visual A->B effect. 3-6 pairs is the
    recommended range.

    ``prompt`` is the conditioning used for both reconstructions. Empty string =
    the bare empty-prompt image-pair slider (pure CS Eq. 9). A non-empty prompt
    turns the pair into a *combined* (prompt-anchored image) example: the visual
    target is still A/B, but it is learned in the context of that prompt -- the
    hook the prompt+image hybrid regime builds on (docs/slider_lora.md S5).

    ``weight`` mixes several pairs with unequal sampling probability.
    """

    uuid: str
    enabled: bool
    before: str
    after: str
    prompt: str
    weight: float

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("enabled", True, bool, False))
        data.append(("before", "", str, False))
        data.append(("after", "", str, False))
        data.append(("prompt", "", str, False))
        data.append(("weight", 1.0, float, False))

        return SliderImagePairConfig(data)
