"""Module provides inference API"""

from fastapi import FastAPI
from src.utils.functions import load_params, get_abs_path, load_pickle
from src.data.functions import clean_features
from src.features.functions import restore_target
from typing import List, Any
from pydantic import BaseModel, validator
import pandas as pd
import lightgbm as lgb


class PredictRequest(BaseModel):
    """Features to make a predictions
    - data: list of object features, first element (row)
     is the feature names
    """

    data: List[List[Any]]

    # check if all inner lists have the same length
    @validator("data")
    def check_dimensionality(cls, v):
        lens = [len(_) for _ in v]
        if min(lens) != max(lens):
            raise ValueError(
                "Each row must have the same length, "
                f"but {min(lens)}-{max(lens)} range is given"
            )
        return v


class PredictResponse(BaseModel):
    """Predictions
    - data: per night price for Airbnb apartment for given objects
    """

    data: List[float]


PARAMS = load_params()

INFO = {
    "title": PARAMS["title"],
    "description": PARAMS["description"],
    "version": PARAMS["version"],
}

app = FastAPI(**INFO)


@app.get("/")
def get_info() -> dict:
    """Returns general information about API:
    - title
    - description
    - version
    """
    return INFO


@app.post("/predict", response_model=PredictResponse)
async def make_predictions(payload: PredictRequest):
    """Predicts per night price for Airbnb apartment for given objects

    Params:
        PredictRequest - list of object features to predict,
            first element shall be feature names
    """

    model_path = get_abs_path(
        PARAMS["model"]["path"],
        PARAMS["model"]["model_file"],
    )
    column_transformer_path = get_abs_path(
        PARAMS["model"]["path"],
        PARAMS["model"]["column_transformer_file"],
    )

    model = lgb.Booster(model_file=model_path)
    column_transformer = load_pickle(column_transformer_path)

    dataset = clean_features(
        pd.DataFrame(payload.data[1:], columns=payload.data[0])
    )

    valid_features = dataset[dataset.is_valid]
    if valid_features.shape[0]:
        features = column_transformer.transform(valid_features)
        predictions = model.predict(features)
        predictions = restore_target(predictions)
        dataset = dataset.join(
            pd.DataFrame(
                {"predictions": predictions}, index=valid_features.index
            )
        )
        dataset.predictions.fillna(-1.0, inplace=True)
    else:
        dataset["predictions"] = -1.0

    return PredictResponse(data=dataset.predictions.tolist())
