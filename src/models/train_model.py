import click
from os import path
from src.utils.functions import (
    load_params,
    get_project_dir,
    setup_logging,
)
import logging
import lightgbm as lgb
import pandas as pd


@click.command()
def main() -> None:
    """Trains model"""

    logger = logging.getLogger(__name__)

    params = load_params()
    train_dataset_path = path.join(
        get_project_dir(),
        params["data"]["processed_data_path"],
        params["data"]["train_dataset_file"],
    )
    logger.info(f"Train model with dataset {train_dataset_path}")

    model_path = path.join(
        get_project_dir(),
        params["model"]["path"],
        "lgbm_regressor.txt",
    )
    eval_hist_path = path.join(
        get_project_dir(),
        params["model"]["eval_hist_path"],
        "lgbm_regressor_eval.csv",
    )

    dataset = lgb.Dataset(train_dataset_path)
    logger.info("Loaded dataset")

    model_params = {
        "task": "train",
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 256,
        "learning_rate": 0.001,
        "feature_fraction": 1.0,
        "min_data_in_leaf": 5,
        "max_depth": 16,
        "seed": params["random_seed"],
        "verbose": 1,
    }
    eval_hist = lgb.cv(
        model_params,
        dataset,
        num_boost_round=10000,
        nfold=5,
        stratified=False,
        shuffle=True,
        callbacks=[lgb.early_stopping(50)],
        return_cvbooster=True,
    )

    cvbooster = eval_hist.pop("cvbooster")
    cvbooster.save_model(model_path)

    pd.DataFrame(eval_hist).to_csv(eval_hist_path)
    logger.info("Model is trained")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
