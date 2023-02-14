"""Provides data cleaning functions"""

import numpy as np
import pandas as pd


def true_false_to_int(value: str) -> int:
    """Transforms t/f values into 1/0"""
    if value == "t":
        return 1
    elif value == "f":
        return 0
    else:
        return np.NaN


def price_to_int(value: str) -> int:
    """Transforms price from string to int"""

    return int(value.replace(",", "")[1:-3])


def is_features_valid(row: pd.DataFrame) -> bool:
    """Validates if feature values are in a valid range"""
    return row.bedrooms < 5 and row.accommodates < 9


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans features"""

    data.host_is_superhost = data.host_is_superhost.apply(true_false_to_int)
    data["is_valid"] = data.apply(is_features_valid, axis=1)
    return data
