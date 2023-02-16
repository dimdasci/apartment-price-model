"""
This module provides application-specific datatypes for
data processing.

The module includes the following classes:
    - `DatasetStage`: An enumeration representing the stages
      where the processed data will be used.

The `DatasetStage` enumeration provides the following members:
    - `TRAIN`: dataset for a model training stage.
    - `TEST`: dataset for a model test stage.
"""

from enum import Enum


class DatasetStage(Enum):
    """
    An enumeration representing the stages of data processing.

    This enumeration provides two members: `TRAIN` and `TEST`.
    These members can be used to label datasets according to the stage
    of model development in which they will be used.

    Attributes:
        - `TRAIN`: Represents a dataset used during the model
          training stage.
        - `TEST`: Represents a dataset used during the model
          testing stage.
    """

    TRAIN = "train"
    TEST = "test"
