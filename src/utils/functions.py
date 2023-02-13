"""Basic utils

Exports:

    - load_params       - loads params from params.yaml in the project root
    - setup_logging     - sets up logging handlers and a format
    - get_project_dir
    - load_pickle
    - save_pickle
"""

import yaml
import logging
from pathlib import Path
import os
import pickle
from typing import Any


def load_params() -> dict:
    """Loads params from params.yaml in the project root"""
    params_path = os.path.join(get_project_dir(), "params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def setup_logging(
    logname: str = "", logfile: str = "logs/app.log", loglevel: str = "DEBUG"
) -> logging.Logger:
    """
    Sets up logging handlers and a format

    Params
        logname: str - name of logger
        logfile: str - path to log file
        loglevel: str - loglevel
    Returns
        Logger
    """
    loglevel = getattr(logging, loglevel)

    logger = logging.getLogger(logname)
    logger.setLevel(loglevel)
    fmt = (
        "%(asctime)s: %(levelname)s: %(filename)s: "
        + "%(funcName)s(): %(lineno)d: %(message)s"
    )
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_project_dir() -> str:
    """Returns project directory"""
    return Path(__file__).resolve().parents[2]


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
