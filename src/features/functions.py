"""
The target_transform module contains functions for transforming and
restoring target variables using log10 transformation.

Functions:
----------
1. transform_target(target: pd.Series) -> pd.Series:
    Applies the log10 transformation to the input target series.

2. restore_target(target: pd.Series) -> pd.Series:
    Restores the original target variable by applying the exponential
    transformation.

Dependencies:
-------------
- pandas
- numpy

Example:
--------
import pandas as pd
from src.features.functions import transform_target, restore_target

# create a sample target variable
target = pd.Series([1, 10, 100])

# transform the target variable
transformed_target = transform_target(target)
print(transformed_target)

# Output:
# 0    0.000000
# 1    1.000000
# 2    2.000000
# dtype: float64

# restore the original target variable
restored_target = restore_target(transformed_target)
print(restored_target)

# Output:
# 0       1.0
# 1      10.0
# 2     100.0
# dtype: float64

"""

import pandas as pd
import numpy as np


def transform_target(target: pd.Series) -> pd.Series:
    """
    Transforms the target variable by applying the log10 transformation.

    Params:
        target: pandas.Series
            The target variable to be transformed.

    Returns:
        pandas.Series
            The transformed target variable.
    """

    return np.log10(target)


def restore_target(target: pd.Series) -> pd.Series:
    """
    Restores the original target variable by applying the exponential
    transformation.

    Params:
        target: pandas.Series
            The transformed target variable to be restored.

    Returns:
        pandas.Series
            The restored target variable.

    """

    return 10**target
