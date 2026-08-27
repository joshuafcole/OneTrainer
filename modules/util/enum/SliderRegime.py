from enum import Enum


class SliderRegime(Enum):
    # Prompt-pair Concept Sliders (Gandikota et al., ECCV 2024, Eq. 7/8): no image
    # dataset. The frozen base supplies the guidance direction v(c+) - v(c-) and
    # x_t is generated on-manifold by partial denoising under the target concept.
    PROMPT_PAIR = 'PROMPT_PAIR'

    def __str__(self):
        return self.value
