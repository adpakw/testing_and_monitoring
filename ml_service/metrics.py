import time

from fastapi import Response
from prometheus_client import (REGISTRY, Counter, Gauge, Histogram,
                               generate_latest)

REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Model inference time",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Probability of positive class",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTION_CLASS = Counter("prediction_class", "Predicted class", ["class"])

PREPROCESS_DURATION = Histogram(
    "preprocess_duration_seconds", "Time to preprocess input features"
)

MODEL_INFO = Gauge(
    "model_info", "Information about current model", ["run_id", "features"]
)
MODEL_UPDATE = Counter("model_updates_total", "Number of model updates")
