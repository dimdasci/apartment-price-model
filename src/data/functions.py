"""
A module for cleaning and transforming feature values for a set of apartments.

This module contains three functions:

- true_false_to_int: Transforms 't' and 'f' values to 1 and 0, respectively.
- is_features_valid: Validates whether feature values are within a valid range.
- clean_features: Cleans and transforms feature values in a Pandas DataFrame
  based on settings in params.yaml.

The module requires the Pandas library to be installed.

Example usage:
    import pandas as pd
    from src.data import functions

    # Load the vacation rental data into a Pandas DataFrame
    data = pd.read_csv('apartments.csv')

    # Clean and transform the feature values
    cleaned_data = functions.clean_features(data)
"""

import numpy as np
import pandas as pd
from src.utils.functions import load_params


def true_false_to_int(value: str) -> float:
    """
    Transforms a string representing a boolean value into a float.

    Params:
        value: A string representing a boolean value, either 't' or 'f'.

    Returns:
        - An integer, either 1 (if value is 't') or 0 (if value is 'f'),
        - otherwise np.nan
    """
    if value == "t":
        return 1.0
    elif value == "f":
        return 0.0
    else:
        return np.NaN


def price_to_int(value: str) -> int:
    """
    Converts a price string to an integer.

    Params:
        value: A string representing a price in US dollars. The string should
               be in the format '$X,XXX.XX', where 'X' represents a digit.

    Returns:
        An integer representing the price in US dollars.
    """

    return int(value.replace(",", "")[1:-3])


def is_features_valid(row: pd.DataFrame) -> bool:
    """
    Check whether a Pandas DataFrame row satisfies the feature limits
    specified in the configuration file.

    Params:
        row: pd.DataFrame
            A Pandas DataFrame row containing feature values to be
            validated.

    Returns:
        bool
            True if all feature values in the row are below their respective
            limits, False otherwise.

    """
    params = load_params()

    return all(
        [
            row[feature] < limit
            for feature, limit in params["data_cleaning"][
                "feature_limits"
            ].items()
        ]
    )


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans and transforms feature values in a Pandas DataFrame.

    Params:
        data: A Pandas DataFrame containing feature values for a set of
            vacation rental properties. The DataFrame should contain the
            following columns:
            - 'host_is_superhost': a boolean value indicating whether
              the host is a superhost (a string value of 't' or 'f').
            - 'bedrooms': the number of bedrooms in the property (an integer).
            - 'accommodates': the maximum number of guests the property can
              accommodate (an integer).

    Returns:
        A new Pandas DataFrame with the following modifications:
        - The 'host_is_superhost' column is converted to an integer value,
          where 't' corresponds to 1 and 'f' corresponds to 0.
        - A new column called 'is_valid' is added to the DataFrame, containing
          a boolean value indicating whether the feature values for each row
          are within a valid range. The 'is_valid' column is computed using the
          'is_features_valid' function.
    """

    data.host_is_superhost = data.host_is_superhost.apply(true_false_to_int)
    data["is_valid"] = data.apply(is_features_valid, axis=1)
    return data
