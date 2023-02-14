import pytest
from src.features.functions import transform_target, restore_target
from src.utils.functions import load_params
import numpy as np

@pytest.mark.parametrize("test_input", ["column_transformer_file", "path"])
def test_model_params(test_input):
    params = load_params()
    assert "model" in params and params["model"].get(test_input) is not None

@pytest.mark.parametrize("test_input", ["interim_data_path", "processed_data_path"])
def test_data_params(test_input):
    params = load_params()
    assert "data" in params and params["data"].get(test_input) is not None

@pytest.mark.parametrize("test_input", np.arange(10, 1000, 100))
def test_target_transform_restore(test_input):
    assert np.isclose(restore_target(transform_target(test_input)), test_input)