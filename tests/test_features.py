import pytest
from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest

def test_to_dataframe_success():
    req = PredictRequest(age=30, workclass="Private")
    df = to_dataframe(req, needed_columns=["age", "workclass"])
    assert df.shape == (1, 2)
    assert list(df.columns) == ["age", "workclass"]

def test_to_dataframe_missing_column():
    req = PredictRequest(age=30, workclass="Private")
    with pytest.raises(ValueError, match="Missing required features: education"):
        to_dataframe(req, needed_columns=["age", "education"])