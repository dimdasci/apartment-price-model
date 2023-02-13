import numpy as np
import click
from enum import Enum
from os import path
from src.utils.functions import load_params, get_project_dir, setup_logging
import logging
import pandas as pd


class Dataset(Enum):
    TRAIN = "train"
    TEST = "test"


def true_false_to_int(value: str) -> int:
    """Transforms t/f values into 1/0"""
    if value == "t":
        return 1
    elif value == "f":
        return 0
    else:
        return np.NaN


def price_to_int(value: str) -> int:
    """Transforms price from string to int"""

    return int(value.replace(",", "")[1:-3])


def is_features_valid(row: pd.DataFrame) -> bool:
    """Validates if feature values are in a valid range"""
    return row.price < 1000 and row.bedrooms < 5 and row.accommodates < 9


@click.command()
@click.option("-d", "--dataset", type=Dataset, help="train or test dataset to clean")
def main(dataset: Dataset) -> None:
    """Cleans features of the dataset"""

    logger = logging.getLogger(__name__)

    params = load_params()
    source_dataset_path = path.join(
        get_project_dir(),
        params["data"]["raw_data_path"],
        params["data"][f"{dataset.value}_data_file"],
    )
    dest_dataset_path = path.join(
        get_project_dir(),
        params["data"]["interim_data_path"],
        params["data"][f"{dataset.value}_data_file"],
    )
    logger.info(f"Clean dataset {source_dataset_path}")

    df = pd.read_csv(source_dataset_path)
    logger.info(f"Loaded dataset shape {df.shape}")

    df.price = df.price.apply(price_to_int)
    df.host_is_superhost = df.host_is_superhost.apply(true_false_to_int)
    df = df.drop_duplicates().dropna(axis=0)
    df = df[df.apply(is_features_valid, axis=1)].query("price < 1000")

    logger.info(f"Cleaned dataset shape {df.shape}")
    df.to_csv(dest_dataset_path, index=False)
    logger.info(f"Cleaning {dataset.value} is done")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
