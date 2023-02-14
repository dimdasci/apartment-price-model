import numpy as np
import click
from src.utils.functions import (
    load_params,
    get_abs_path,
    setup_logging,
    load_pickle,
    save_pickle,
)
from src.data.datatypes import DatasetStage
from src.features.functions import transform_target
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@click.command()
@click.option(
    "-s", "--stage", type=DatasetStage, help="train or test dataset to clean"
)
def main(stage: DatasetStage) -> None:
    """Cleans features of the dataset depending on stage train/test"""

    logger = logging.getLogger(__name__)

    params = load_params()
    column_transformer_path = get_abs_path(
        params["model"]["path"],
        params["model"]["column_transformer_file"],
    )
    source_dataset_path = get_abs_path(
        params["data"]["interim_data_path"],
        params["data"][f"{stage.value}_data_file"],
    )
    dest_dataset_path = get_abs_path(
        params["data"]["processed_data_path"],
        params["data"][f"{stage.value}_data_file"],
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
        # we initialize and fit transformer to encode categorical
        # features with Ordinal Encoder and scale numerical
        # with StandardScaler
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
        # save transformer for test dataset processing stage
        save_pickle(column_transformer, column_transformer_path)
        logger.info("Saved fitted column transformer")

        # save categorical feature names
        pd.DataFrame({"categorical": categorical_features}).to_csv(
            get_abs_path(
                params["data"]["processed_data_path"],
                params["data"]["categorical_feature_names_file"],
            ),
            index=False,
        )

    else:
        # load fitted transformer
        column_transformer = load_pickle(column_transformer_path)
        logger.info("Loaded fitted column transformer")

    features_transformed = column_transformer.transform(features)
    target_transformed = transform_target(target)
    logger.info(
        f"Transformed {features_transformed.shape} features and "
        f"{target_transformed.shape} target"
    )

    dataset = pd.DataFrame(
        features_transformed, columns=categorical_features + numerical_features
    ).join(target_transformed)

    dataset.to_csv(dest_dataset_path, index=False)


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
