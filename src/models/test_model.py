"""
This module provides a command-line interface for testing a saved LightGBM
model against a test dataset and calculates the R2, MAE, MAPE, and RMSE
performance metrics, logs them, and saves them to a CSV file.

The dataset and the paths to the model and model performance files
are specified in a config file `params.yaml` which is loaded with the
`load_params()` function.

This module provides a `main()` function that can be run as a command line
interface.

Usage:
    $ python test_model.py

Returns:
    None
"""


import click
from src.utils.functions import (
    load_params,
    get_abs_path,
    setup_logging,
)
import logging
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)


@click.command()
def main() -> None:
    """Tests model

    This function loads a saved LightGBM model and tests it against
    a test dataset. The dataset and the paths to the model and model
    performance files are specified in a config file `params.yaml`
    which is loaded with the `load_params()` function.

    The function calculates the R2, MAE, MAPE, and RMSE performance metrics,
    logs them, and saves them to a CSV file at the location specified in
    the config file.

    Returns:
        None: The function doesn't return anything.
    """

    logger = logging.getLogger(__name__)

    params = load_params()
    test_dataset_path = get_abs_path(
        params["data"]["processed_data_path"],
        params["data"]["test_data_file"],
    )
    logger.info(f"Test model with dataset {test_dataset_path}")

    model_path = get_abs_path(
        params["model"]["path"],
        params["model"]["model_file"],
    )
    model_performance_path = get_abs_path(
        params["model"]["report_path"],
        params["model"]["model_performance_file"],
    )

    df = pd.read_csv(test_dataset_path)
    features = df[params["data"]["features"]]
    target = df[params["data"]["target"]]

    model = lgb.Booster(model_file=model_path)

    preds = model.predict(features.values)
    metrics = {
        "r2": [r2_score(target, preds)],
        "mae": [mean_absolute_error(10**target, 10**preds)],
        "mape": [mean_absolute_percentage_error(10**target, 10**preds)],
        "rmse": [mean_squared_error(10**target, 10**preds) ** 0.5],
    }
    logger.info(f"R2:   {np.mean(metrics['r2']):>8.4f}")
    logger.info(f"MAE:  {np.mean(metrics['mae']):>8.4f}")
    logger.info(f"MAPE: {np.mean(metrics['mape']):>8.4f}")
    logger.info(f"RMSE: {np.mean(metrics['rmse']):>8.4f}")

    pd.DataFrame(metrics).to_csv(model_performance_path, index=False)
    logger.info("Done model test")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
