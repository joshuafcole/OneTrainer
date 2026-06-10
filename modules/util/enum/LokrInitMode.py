from enum import Enum


class LokrInitMode(Enum):
    DEFAULT = 'DEFAULT'
    GRADIENT = 'GRADIENT'

    def __str__(self):
        return self.value
