from src.data.functions import (true_false_to_int, price_to_int, is_features_valid)
from src.utils.functions import load_params
import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize("test_input,expected", [("f", 0), ("t", 1)])
def test_true_false_to_int_valid(test_input, expected):
    assert true_false_to_int(test_input) == expected

def test_true_false_to_int_invalid():
    assert np.isnan(true_false_to_int("F"))

@pytest.mark.parametrize("test_input,expected", [("$12,456.00", 12456), ("$2.00", 2)])
def test_price_to_int(test_input, expected):
    assert price_to_int(test_input) == expected

def test_is_features_valid():
    df = pd.DataFrame({
        'bedrooms': [3, 6, 4, 8, 3], 
        'accommodates': [5, 5, 12, 10, 10],
        'beds': [3, 3, 8, 3, 3]
        })
    assert df.apply(is_features_valid, axis=1).to_list() == [True, False, False, False, True]

@pytest.mark.parametrize("test_input", ["source_url", "features", "target", 
                                        "test_split_ratio", "train_data_file", "test_data_file"])
def test_params(test_input):
    params = load_params()
    assert "data" in params and params["data"].get(test_input) is not None