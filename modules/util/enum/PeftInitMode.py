from enum import Enum


class PeftInitMode(Enum):
    DEFAULT = 'DEFAULT'
    GRADIENT = 'GRADIENT'

    def __str__(self):
        return self.value
