# Apartment price model

[![Build Test](https://github.com/dimdasci/apartment-price-model/actions/workflows/makefile.yaml/badge.svg)](https://github.com/dimdasci/apartment-price-model/actions/workflows/makefile.yaml)

In the project, we show a fast and reproducible way to build and deploy a machine-learning model. Our pricing model predicts Airbnb apartment prices based on its properties and amenities. The model is available in a containerized API and trained using http://insideairbnb.com dataset, with an automated data processing and training pipeline


## Motivation

Recognizing that data science is an iterative process with a high degree of uncertainty, the ultimate goal of a data science team is to deliver value to an organization quickly and reliably.

The goal of this data science project is to provide a fast, reliable, and reproducible solution that can automate end-to-end processes from acquiring raw data to providing an API endpoint for predicting an acceptable price per night for Airbnb apartments.

- The project will provide an environment that supports both Jupyter notebooks and Python CLI commands for automation at every stage of data processing and model training.
- A CI workflow will be implemented to test the code on pull requests.
- The project configuration will localize all settings in a single place.


In summary, the proposed approach to organizing and executing a data science project allows us to provide a ready-made solution within a couple of working days that can be integrated into end-user products.

The whole team will benefit from discovering the power of machine learning from the very beginning of the project, and the data science team from getting quick feedback across the spectrum of model use. This will reduce the uncertainty inherent in data science projects, leaving only iterations to improve data understanding and model quality.


## Approach

The proposed approach to the solution includes the following steps: 
- Take the minimum set of features, based on common sense. 
- Perform an initial explanatory analysis of the data to understand the overall structure of the data and identify critical issues that are blocking the model development. 
- Correct these issues, train the model and evaluate its performance on the test dataset. 
- Deliver the trained model to an API-wrapped Docker container with a prediction endpoint. 
- The project is ready for continuous improvement until the quality of the model meets the business goals.

The project is based on cookiecutter data science template and was done in the following steps.
1. CLI command `src/data/make_dataset.py`:
 - Downloads a dataset from a source URL specified in the configuration file.
 - Selects specified in the configuration features and target columns from the downloaded dataset.
 - Splits the dataset into training and test subsets.
 - Saves the resulting datasets to the raw data path specified in the configuration file.
1. Initial EDA
2. CLI command `src/data/clean_data.py`
 - Drops duplicates and null values. 
 - Converts the target column to an integer.
 - Applies custom feature cleaning function.
 - Filters out invalid rows.
 - Saves the cleaned dataset to the interim data path.
3. CLI command `src/features/build_features.py`
 - Splits the features into categorical and numerical features.
 - Initializes and fits a transformer to encode the categorical features with Ordinal Encoder and scale the numerical features with StandardScaler for the training dataset.
 - Transforms the features and target using the fitted transformer.
 - Joins the transformed features and target column into a new dataset.
 - Saves the new dataset to the processed data path. 
4. CLI command `src/model/train_model.py`
 - Reads the training dataset and categorical feature names from CSV files.
 - Trains a LightGBM model with cross-validation and early stopping.
 - Saves the trained model and evaluation history to models and reports data paths accordingly.
5. CLI command `src/model/test_model.py`
 - Loads a trained LightGBM model. 
 - Tests it against a test dataset.
 - Calculates the R2, MAE, MAPE, and RMSE performance metrics, logs them, and saves them to reports data path as a CSV file.
6. API with `predict` endpoint in Docker `src/api`
 - Checks if feature values fit the rules established on EDA and data cleaning stages, otherwise returns a -1 value instead of prediction for invalid entries.
 - Transforms features and target.
 - Load saved model. 
 - Makes predictions.
 - Transforms prediction.
 - Returns predictions.
7. Tests
8. CI workflow
9. Documentation




## Usage


### Setup Project
- clone repo `git clone git@github.com:dimdasci/apartment-price-model.git`
- change directory `cd apartment-price-model`
- create virtual environment `make create_environment`
- activate it `conda activate apartment_price_model`
- run installation scripts `make install` to prepare the project for running

### Project configuration

The project configuration is specified in the `params.yaml` file.


### Run pipeline

To run the pipeline from getting data to delivering trained and tested model use the command

`make pipeline`

The command goes through the following steps, which can be performed independently with `make` command, for example: `make get_data`.
- `get data` — downloads data set, splits it on training and test subsets,
- `clean_data` — fixes data types, removes duplicates and manages missing values. It requires option `--stage` to specify the dataset to clean (`train` or `test`), 
- `build_features` — transforms columns to featuresIt requires option `--stage` to specify the dataset to transform (`train` or `test`),
- `train_model` — runs model training with cross-validation,
- `test_model` — tests model on test dataset.

### Run inference API

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
