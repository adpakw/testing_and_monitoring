from fastapi.testclient import TestClient
from ml_service.app import app
import mlflow

client = TestClient(app)

def test_health_no_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert "run_id" in response.json()

def test_predict_without_model():
    response = client.post("/predict", json={"age": 30})
    assert response.status_code == 503

def test_update_model_invalid_run():
    response = client.post("/updateModel", json={"run_id": "invalid"})
    assert response.status_code == 400

def test_predict_after_update(mocker):
    dummy_model = mocker.Mock()
    dummy_model.predict_proba.return_value = [[0.2, 0.8]]
    mocker.patch("mlflow.sklearn.load_model", return_value=dummy_model)
    dummy_model.feature_names_in_ = ["age"]

    response = client.post("/updateModel", json={"run_id": "some_run"})
    assert response.status_code == 200
    response = client.post("/predict", json={"age": 30})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probability"] == 0.8