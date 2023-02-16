"""
This module provides a command-line interface for preparing training
and test datasets.

The module includes the following functions:
    - `main()`: Runs the command-line interface for preparing
      training and test datasets.

The module can be run as a standalone program, in which case the `main()`
function is executed with default settings.

Usage:
    $ python make_dataset.py

Returns:
    None

Side effects:
    - Downloads a dataset from a source URL specified in the
      configuration file
    - Selects specific features and a target variable from the
      downloaded dataset
    - Splits the dataset into training and test subsets
    - Saves the resulting datasets to the raw data path specified
      in the configuration file
    - Writes log messages to a file named after the module

The module reads the configuration from `params.yaml` int the project
directory. The configuration file is expected to be in YAML format,
and must contain the following keys:

    - "data": a dictionary containing the following keys:
        - "source_url": a string specifying the URL of the dataset to
          download
        - "features": a list of strings specifying the names of the columns
          to include as features
        - "target": a string specifying the name of the column to use as the
          target variable
        - "test_split_ratio": a float between 0 and 1 specifying the proportion
          of the data to use for testing
        - "raw_data_path": a string specifying the directory where the raw data
          files should be saved
        - "train_data_file": a string specifying the filename to use for the
          training data file
        - "test_data_file": a string specifying the filename to use for the
          test data file
    - "random_seed": an integer specifying the random seed to use for the
      train-test split

The log messages are written to a `log/app.log` file in the project directory.
The log level is set to INFO by default, but can be customized by passing a
different value for the `loglevel` argument of the `setup_logging()` function.
"""


import click
from src.utils.functions import load_params, setup_logging, get_abs_path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
def main() -> None:
    """
    Runs the main command-line interface (CLI) function for preparing
    training and test datasets.

    Returns:
        None

    Raises:
        AssertionError: If the downloaded dataset has no rows or columns

    Side effects:
        - Downloads a dataset from the source URL specified in the
          configuration file
        - Selects specific features and a target variable from the
          downloaded dataset
        - Splits the dataset into training and test subsets
        - Saves the resulting datasets to the raw data path specified
          in the configuration file

    The function expects a valid configuration file to be present in the
    project directory.
    """

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
