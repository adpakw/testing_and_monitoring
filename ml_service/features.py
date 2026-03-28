import pandas as pd
from ml_service.schemas import PredictRequest

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = needed_columns or FEATURE_COLUMNS
    req_dict = req.dict(by_alias=True)
    missing = [col for col in columns if col not in req_dict or req_dict[col] is None]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = [req_dict[col] for col in columns]
    return pd.DataFrame([row], columns=columns)
