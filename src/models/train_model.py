import click
from src.utils.functions import (
    load_params,
    get_abs_path,
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
    train_dataset_path = get_abs_path(
        params["data"]["processed_data_path"],
        params["data"]["train_data_file"],
    )
    logger.info(f"Train model with dataset {train_dataset_path}")

    model_path = get_abs_path(
        params["model"]["path"],
        params["model"]["model_file"],
    )
    eval_hist_path = get_abs_path(
        params["model"]["report_path"],
        params["model"]["eval_hist_file"],
    )
    categorical_features_path = get_abs_path(
        params["data"]["processed_data_path"],
        params["data"]["categorical_feature_names_file"],
    )

    categorical_features = pd.read_csv(
        categorical_features_path
    ).categorical.to_list()
    logger.info(f"Categorical feature names {', '.join(categorical_features)}")

    df = pd.read_csv(train_dataset_path)
    features = df[params["data"]["features"]]
    target = df[params["data"]["target"]]

    dataset = lgb.Dataset(
        features,
        label=target,
        feature_name=features.columns.to_list(),
        categorical_feature=categorical_features,
        free_raw_data=False,
    ).construct()

    logger.info(
        f"Constructed dataset with {dataset.num_data()} rows "
        f"and {dataset.num_feature()} features"
    )

    model_params = {
        "task": "train",
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.001,
        "lambda_l2": 0.5,
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
