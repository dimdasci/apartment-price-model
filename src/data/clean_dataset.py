"""
This module provides a script for cleaning and preprocessing a dataset
for machine learning training or testing.

The module includes the following functions:
    - `main()`: Cleans the features of a dataset depending on the specified
      stage (train or test) by dropping duplicates and null values, converting
      the 'price' column to integer, applying custom feature cleaning
      functions, filtering out invalid rows, and saving the cleaned dataset
      to the interim data path.

Usage:
To clean the train dataset, run the script with the '-s' or '--stage'
option set to 'train':

    python clean_dataset.py -s train

To clean the test dataset, run the script with the '-s' or '--stage'
option set to 'test':

    python clean_dataset.py -s test
"""

import click
from os import path
from src.utils.functions import load_params, get_project_dir, setup_logging
from src.data.datatypes import DatasetStage
from src.data.functions import (
    price_to_int,
    clean_features,
)
import logging
import pandas as pd


@click.command()
@click.option(
    "-s", "--stage", type=DatasetStage, help="train or test dataset to clean"
)
def main(stage: DatasetStage) -> None:
    """
    Cleans features of the specified dataset stage (train or test) by dropping
    duplicates and null values, converting the 'price' column to integer,
    applying custom feature cleaning functions, filtering out invalid rows,
    and saving the cleaned dataset to the interim data path.

    Params:
        stage (DatasetStage): The dataset stage to clean. Must be a
                              value of the DatasetStage enum.

    Returns:
        None.
    """

    logger = logging.getLogger(__name__)

    params = load_params()
    source_dataset_path = path.join(
        get_project_dir(),
        params["data"]["raw_data_path"],
        params["data"][f"{stage.value}_data_file"],
    )
    dest_dataset_path = path.join(
        get_project_dir(),
        params["data"]["interim_data_path"],
        params["data"][f"{stage.value}_data_file"],
    )
    logger.info(f"Clean dataset {source_dataset_path}")

    df = pd.read_csv(source_dataset_path)
    logger.info(f"Loaded dataset shape {df.shape}")

    # drop duplicates and rows with missing values
    df = df.drop_duplicates().dropna(axis=0)

    # convert price from string to int
    df.price = df.price.apply(price_to_int)

    # clean features
    df = clean_features(df)

    # filter valid rows with price less than 1000 and then drop is_valid column
    df = df[df.is_valid].query("price < 1000").drop("is_valid", axis=1)

    logger.info(f"Cleaned dataset shape {df.shape}")
    df.to_csv(dest_dataset_path, index=False)
    logger.info(f"Cleaning {stage.value} is done")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
