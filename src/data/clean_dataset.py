import numpy as np
import click
from os import path
from src.utils.functions import load_params, get_project_dir, setup_logging
from src.data.datatypes import DatasetStage
from src.data.functions import (
    true_false_to_int,
    price_to_int,
    is_features_valid,
    clean_features,
)
import logging
import pandas as pd


@click.command()
@click.option(
    "-s", "--stage", type=DatasetStage, help="train or test dataset to clean"
)
def main(stage: DatasetStage) -> None:
    """Cleans features of the dataset depending on stage train/test"""

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

    df = df.drop_duplicates().dropna(axis=0)
    df.price = df.price.apply(price_to_int)
    df = clean_features(df)
    df = df[df.is_valid].query("price < 1000").drop("is_valid", axis=1)

    logger.info(f"Cleaned dataset shape {df.shape}")
    df.to_csv(dest_dataset_path, index=False)
    logger.info(f"Cleaning {stage.value} is done")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
