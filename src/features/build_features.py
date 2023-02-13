import numpy as np
import click
from os import path
from src.utils.functions import (
    load_params,
    get_project_dir,
    setup_logging,
    load_pickle,
    save_pickle,
)
from src.data.datatypes import DatasetStage
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import lightgbm as lgb


@click.command()
@click.option(
    "-s", "--stage", type=DatasetStage, help="train or test dataset to clean"
)
def main(stage: DatasetStage) -> None:
    """Cleans features of the dataset depending on stage train/test"""

    logger = logging.getLogger(__name__)

    params = load_params()
    column_transformer_path = path.join(
        get_project_dir(),
        params["model"]["path"],
        "column_transformer.pkl",
    )
    source_dataset_path = path.join(
        get_project_dir(),
        params["data"]["interim_data_path"],
        params["data"][f"{stage.value}_data_file"],
    )
    dest_dataset_path = path.join(
        get_project_dir(),
        params["data"]["processed_data_path"],
        params["data"][f"{stage.value}_dataset_file"],
    )
    logger.info(f"Build features for dataset {source_dataset_path}")

    df = pd.read_csv(source_dataset_path)
    logger.info(f"Loaded dataset shape {df.shape}")

    features = df[params["data"]["features"]]
    target = df[params["data"]["target"]]

    categorical_features = features.select_dtypes(
        exclude="number"
    ).columns.to_list()
    numerical_features = features.select_dtypes(
        include="number"
    ).columns.to_list()
    logger.info(f"Categorical features {', '.join(categorical_features)}")
    logger.info(f"Numerical features {', '.join(numerical_features)}")
    logger.debug(f"num of categorical features {len(categorical_features)}")
    logger.debug(f"num of numerical features {len(numerical_features)}")
    assert (
        len(categorical_features) + len(numerical_features)
        == features.shape[1]
    )

    if stage == DatasetStage.TRAIN:
        column_transformer = ColumnTransformer(
            [
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                    ),
                    categorical_features,
                ),
                ("numerical", StandardScaler(), numerical_features),
            ],
            remainder="drop",
        )
        column_transformer.fit(features)
        save_pickle(column_transformer, column_transformer_path)
        logger.info("Saved fitted column transformer")

    else:
        column_transformer = load_pickle(column_transformer_path)
        logger.info("Loaded fitted column transformer")

    features_transformed = column_transformer.transform(features)
    target_transformed = np.log10(target)
    logger.info(
        f"Transformed {features_transformed.shape} features and "
        f"{target_transformed.shape} target"
    )

    dataset = lgb.Dataset(
        features_transformed,
        label=target_transformed,
        feature_name=categorical_features + numerical_features,
        categorical_feature=categorical_features,
    )
    dataset.save_binary(dest_dataset_path)


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="DEBUG")

    main()
