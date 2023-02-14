# Apartment price model
==============================

[![Build Test](https://github.com/dimdasci/apartment-price-model/actions/workflows/makefile.yml/badge.svg)](https://github.com/dimdasci/apartment-price-model/actions/workflows/makefile.yml)

A pricing model, which can predict the acceptable per night price for Airbnb apartment based on its properties and the offered amenities

## Setup Project
- clone repo `git clone git@github.com:dimdasci/apartment-price-model.git`
- change directory `cd apartment-price-model`
- create virtual environment `make create_environment`
- activate it `conda activate apartment_price_model`
- run installation scripts `make install` to test development environment

## Project configuration

The project configuration is specified in the `params.yaml` file.


## Train model

To train model run 

`make pipeline`

The command goes through the following steps, which can be performed independently with `make` command, for example: `make get_data`.
- `get data` — downloads data set, splits it on training and test subsets,
- `clean_data` — fixes data types, removes duplicates and manages missing values. It requires option `--stage` to specify the dataset to clean (`train` or `test`), 
- `build_features` — transforms columns to featuresIt requires option `--stage` to specify the dataset to transform (`train` or `test`),
- `train_model` — runs model training with cross-validation,
- `test_model` — tests model on test dataset.

## Run inference API

Inference API works in docker container. 

To build the image run

`make build_api`

To run container 

`make run_api`

The API will be available on http://0.0.0.0:8000. API docs http://0.0.0.0:8000/docs.

To get predictions for dataframe `features` use the following example.

```
import requests

url = "http://localhost:8000/predict"
headers = {"content-type": "application/json"}
payload = {
    'data': [features.columns.to_list()] + features.values.tolist()
}

response = requests.post(url, json=payload, headers=headers)
```

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── params.yaml        <- Project configuration.
    ├── Dockerfile         <- Inference API Image.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- Inference API
    │   │   └── main.py
    │   ├── data           <- Scripts to download, transform or generate data
    │   │   ├── make_dataset.py
    │   │   ├── clean_dataset.py
    │   │   ├── datatypes.py
    │   │   └── functions.py
    │   │   
    │   ├── features       <- Scripts to turn data into features for modeling
    │   │   ├── functions.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train and test models               
    │   │   ├── test_model.py
    │   │   └── train_model.py
    │   │
    │   └── utils          <- Scripts with helper functions            
    │       └── functions.py
    │
    ├── tests              <- Tests
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
