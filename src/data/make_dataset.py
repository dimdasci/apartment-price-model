"""Downloads dataset from the URL, splits it into training and
    test datasets, stores them in project directories.
    Takes parameters form params.yaml.
"""

import click
from src.utils.functions import load_params, setup_logging, get_abs_path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
def main() -> None:
    """Main CLI function"""
    logger = logging.getLogger(__name__)

    params = load_params()
    logger.info(f'Getting data from {params["data"]["source_url"]}')

    df = pd.read_csv(params["data"]["source_url"], compression="gzip")
    logger.info(f"Downloaded dataset with {df.shape} shape")
    assert (
        df.shape[0] > 0 and df.shape[1] > 1
    ), f"Downloaded dataset has shape {df.shape}"

    features = params["data"]["features"]
    target = params["data"]["target"]
    logger.info(f'Select features {", ".join(features)} and target {target}')

    train, test = train_test_split(
        df[features + [target]],
        test_size=params["data"]["test_split_ratio"],
        random_state=params["random_seed"],
    )
    logger.info(
        f"Split to training {train.shape} and " f"test {test.shape} subsets"
    )

    train.to_csv(
        get_abs_path(
            params["data"]["raw_data_path"], params["data"]["train_data_file"]
        ),
        index=False,
    )
    test.to_csv(
        get_abs_path(
            params["data"]["raw_data_path"], params["data"]["test_data_file"]
        ),
        index=False,
    )

    logger.info("Training and test datasets are ready")


if __name__ == "__main__":
    logger = setup_logging(logname=__name__, loglevel="INFO")

    main()
