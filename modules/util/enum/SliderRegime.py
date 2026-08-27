from enum import Enum


class SliderRegime(Enum):
    # Prompt-pair Concept Sliders (Gandikota et al., ECCV 2024, Eq. 7/8): no image
    # dataset. The frozen base supplies the guidance direction v(c+) - v(c-) and
    # x_t is generated on-manifold by partial denoising under the target concept.
    PROMPT_PAIR = 'PROMPT_PAIR'

    # Coordinate-labeled image sliders: an ordinary image dataset whose captions
    # carry a declared-axis coordinate token, e.g. "(distance:-2)". The coordinate
    # is stripped from the caption before tokenization -- so the conditioning stays
    # orthogonal to the axis -- and becomes the training-time adapter multiplier
    # m = gain_k * value. The adapter at m must reconstruct that image. Binary
    # before/after poles are the value in {-1, +1} special case, which is why this
    # subsumes an explicit image-pair regime rather than sitting beside one.
    IMAGE = 'IMAGE'

    def __str__(self):
        return self.value
