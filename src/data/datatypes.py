"""Module implements application datatypes"""

from enum import Enum


class DatasetStage(Enum):
    """Stages of model development"""

    TRAIN = "train"
    TEST = "test"
