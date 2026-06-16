from enum import Enum


class SliderRegime(Enum):
    # Prompt-pair Concept Sliders (Gandikota et al., Eq. 7/8): no image dataset;
    # the frozen base supplies the guidance direction v(c+) - v(c-) and x_t is
    # generated on-manifold by partial flow-matching denoising.
    PROMPT_PAIR = 'PROMPT_PAIR'

    # Coordinate-labeled image sliders (docs §10): a real image dataset whose
    # captions carry a declared-axis coordinate token, e.g. "(distance:-2)". The
    # coordinate is stripped from the conditioning and becomes the training-time
    # adapter multiplier m = k*value (coordinate-scaled reconstruction, CS Eq. 9
    # generalized). Binary before/after poles are the value in {-1, +1} case.
    IMAGE = 'IMAGE'

    def __str__(self):
        return self.value
