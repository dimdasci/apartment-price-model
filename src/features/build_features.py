"""
Builds features from a cleaned dataset based on whether the input dataset
is for training or testing.

The module defines a `main` function that performs the following steps:
1. Reads in the input dataset.
2. Extracts the features and target columns from the input dataset.
3. Splits the features into categorical and numerical features.
4. Initializes and fits a transformer to encode the categorical features
   with Ordinal Encoder and scale the numerical features with StandardScaler
   for the training dataset.
5. Transforms the features and target using the fitted transformer.
6. Joins the transformed features and target into a new dataset.
7. Saves the new dataset to a file.

If the dataset is for training, a column transformer is initialized and fitted
to encode categorical features with Ordinal Encoder and scale numerical
features with StandardScaler. The fitted column transformer is saved, and the
categorical feature names are written to a CSV file. If the dataset is for
testing, the saved column transformer is loaded.

The transformed features and the original target are merged into a new
pandas dataframe, which is saved to a CSV file at the destination dataset path.

Modules imported:
- `numpy` for scientific computing with Python.
- `click` for creating command line interfaces in a composable way.
- `pandas` for data manipulation and analysis.
- `logging` for logging events and errors.
- `sklearn` for machine learning algorithms and tools.

Args:
    stage (DatasetStage): Enum representing the stage of the dataset,
                          either training or testing.

Returns:
    None.

Example:
    To build the features for the training dataset, run the following
    command in the terminal:
    ```
    python build_features.py --stage train
    ```

    To build the features for the testing dataset, run the following
    command in the terminal:
    ```
    python build_features.py --stage test
    ```
"""

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
    """
    Builds the features from a cleaned dataset based on whether the input
    dataset is for training or testing.

    Params:
        stage (DatasetStage): Enum representing the stage of the dataset,
                              either training or testing.

    Returns:
        None.

    The function performs the following steps:
    1. Reads in the dataset.
    2. Extracts the features and target columns from the dataset.
    3. Splits the features into categorical and numerical features.
    4. Initializes and fits a transformer to encode the categorical features
       with Ordinal Encoder and scale the numerical features with
       StandardScaler for the training dataset.
    5. Transforms the features and target using the fitted transformer.
    6. Joins the transformed features and target into a new dataset.
    7. Saves the new dataset to a file.

    If the dataset is for training, a column transformer is initialized and
    fitted to encode categorical features with Ordinal Encoder and scale
    numerical features with StandardScaler. The fitted column transformer
    is saved, and the categorical feature names are written to a CSV file.
    If the dataset is for testing, the saved column transformer is loaded.

    The features and target are transformed, and a new pandas dataframe
    is created from the transformed features and the original target.
    The dataset with features is saved to a CSV file at the destination
    dataset path.
    """

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
