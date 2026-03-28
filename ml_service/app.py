import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from ml_service import config
from ml_service.features import to_dataframe
from ml_service.metrics import REQUEST_DURATION, REQUESTS
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from starlette.middleware.base import BaseHTTPMiddleware


MODEL = Model()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_DURATION.labels(
            method=request.method, endpoint=request.url.path
        ).observe(duration)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    try:
        MODEL.set(run_id=run_id)
    except Exception as e:
        # Log the error but keep the service alive
        print(f"Startup model load failed: {e}")
    yield
    # add any teardown logic here if needed


def create_app() -> FastAPI:
    app = FastAPI(title="MLflow FastAPI service", version="1.0.0", lifespan=lifespan)
    app.add_middleware(MetricsMiddleware)

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {"status": "ok", "run_id": run_id}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_state = MODEL.get()
        if model_state.model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
        start_pre = time.time()

        try:
            df = to_dataframe(request, needed_columns=MODEL.features)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        PREPROCESS_DURATION.observe(time.time() - start_pre)

        start_inf = time.time()
        probability = model_state.model.predict_proba(df)[0][1]
        INFERENCE_DURATION.observe(time.time() - start_inf)

        PREDICTION_PROBABILITY.observe(probability)

        prediction = int(probability >= 0.5)
        PREDICTION_CLASS.labels(class=str(prediction)).inc()

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        try:
            MODEL.set(run_id=req.run_id)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Update metrics
        model_state = MODEL.get()
        MODEL_INFO.labels(run_id=model_state.run_id, features=",".join(model_state.model.feature_names_in_)).set(1)
        MODEL_UPDATE.inc()
        return UpdateModelResponse(run_id=req.run_id)

    @app.get('/metrics')
    def metrics():
        return Response(generate_latest(REGISTRY), media_type='text/plain')


app = create_app()
