from enum import Enum


class SliderRegime(Enum):
    # Prompt-pair Concept Sliders (Gandikota et al., Eq. 7/8): no image dataset;
    # the frozen base supplies the guidance direction v(c+) - v(c-) and x_t is
    # generated on-manifold by partial flow-matching denoising.
    PROMPT_PAIR = 'PROMPT_PAIR'

    # Image-pair / visual sliders (Eq. 9): before/after image pairs supply the
    # target velocity; x_t is the noised real latent under an empty prompt.
    IMAGE_PAIR = 'IMAGE_PAIR'

    def __str__(self):
        return self.value
