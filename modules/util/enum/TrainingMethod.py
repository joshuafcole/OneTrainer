from enum import Enum


class TrainingMethod(Enum):
    FINE_TUNE = 'FINE_TUNE'
    LORA = 'LORA'
    EMBEDDING = 'EMBEDDING'
    FINE_TUNE_VAE = 'FINE_TUNE_VAE'
    # Concept Sliders: a LoRA/LoKr adapter trained so that its signed multiplier
    # is a continuous control knob. Offered only for model types with a registered
    # slider setup -- see create.supported_training_methods().
    SLIDER = 'SLIDER'

    def __str__(self):
        return self.value
