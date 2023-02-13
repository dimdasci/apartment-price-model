# -*- coding: utf-8 -*-
import click
from src.utils.functions import load_params, setup_logging
import logging
from pathlib import Path


@click.command()
def main():  # input_filepath, output_filepath
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    logger = setup_logging(logname=__name__, loglevel="DEBUG")
    params = load_params()

    logger.info(f"Running {__name__}")
    logger.info(f"Project dir {project_dir}")
    logger.info(f'data source {params["data"]["source_url"]}')

    main()
