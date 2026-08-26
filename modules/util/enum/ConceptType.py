from enum import Enum


class ConceptType(Enum):
    STANDARD = 'STANDARD'
    VALIDATION = 'VALIDATION'
    PRIOR_PREDICTION = 'PRIOR_PREDICTION'
    # A close-but-wrong image: trained *away* from, through the bounded
    # reference-anchored repulsion in modules/util/loss/counterexample_loss.py.
    # Like PRIOR_PREDICTION it needs the frozen reference forward, so it is
    # LoRA-only (BaseModelSetup.prior_model raises otherwise).
    COUNTEREXAMPLE = 'COUNTEREXAMPLE'

    def __str__(self):
        return self.value
