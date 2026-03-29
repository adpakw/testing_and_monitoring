import logging
import time
from typing import Any, Mapping

import psutil
from fastapi import Request
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info, generate_latest

LOGGER = logging.getLogger(__name__)

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
http_errors_total = Counter(
    "http_errors_total", "Total HTTP errors (status >=400)", ["path", "status"]
)

process_memory_bytes = Gauge("process_memory_bytes", "Process memory usage in bytes")
process_cpu_percent = Gauge("process_cpu_percent", "Process CPU usage percent")

feature_numeric_value = Gauge(
    "feature_numeric_value", "Current numeric feature value", ["feature"]
)
feature_categorical_total = Counter(
    "feature_categorical_total",
    "Count of categorical feature values",
    ["feature", "category"],
)

preprocessing_duration = Histogram(
    "preprocessing_duration_seconds", "Preprocessing duration in seconds"
)
inference_duration = Histogram(
    "inference_duration_seconds", "Model inference duration in seconds"
)

model_probability = Gauge("model_probability", "Predicted probability for class 1")
model_predictions_total = Counter(
    "model_predictions_total", "Number of predictions", ["prediction"]
)

model_updates_total = Counter(
    "model_updates_total", "Number of model updates", ["status"]
)
model_active = Info("model_active", "Currently active model info")
model_features_count = Gauge(
    "model_features_count", "Number of features required by the active model"
)
model_required_feature = Gauge(
    "model_required_feature",
    "Indicates if a feature is required by the model",
    ["feature"],
)


def observe_feature_values(feature_values: Mapping[str, Any]) -> None:
    for feature, value in feature_values.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            feature_numeric_value.labels(feature=feature).set(float(value))
        else:
            category = str(value)
            feature_categorical_total.labels(feature=feature, category=category).inc()


def observe_preprocessing_duration(seconds: float) -> None:
    preprocessing_duration.observe(seconds)


def observe_inference_duration(seconds: float) -> None:
    inference_duration.observe(seconds)


def observe_prediction(probability: float, prediction: int) -> None:
    model_probability.set(probability)
    model_predictions_total.labels(prediction=str(prediction)).inc()


def observe_model_update(
    run_id: str,
    status: str,
    features: list[str] | None = None,
    model_type: str | None = None,
) -> None:
    model_updates_total.labels(status=status).inc()
    if status == "success":
        model_active.info({"run_id": run_id, "type": model_type or "unknown"})
        model_features_count.set(float(len(features or [])))
        for feature in features or []:
            model_required_feature.labels(feature=feature).set(1)


def refresh_resource_metrics() -> None:
    try:
        process = psutil.Process()
        process_memory_bytes.set(float(process.memory_info().rss))
        process_cpu_percent.set(float(process.cpu_percent(interval=None)))
    except Exception:
        LOGGER.exception("Failed to collect process metrics")


async def track_http_metrics(request: Request, call_next):
    method = request.method
    path = request.url.path
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        http_request_duration.labels(method=method, path=path).observe(duration)
        http_requests_total.labels(method=method, path=path, status="500").inc()
        http_errors_total.labels(path=path, status="500").inc()
        refresh_resource_metrics()
        raise

    duration = time.perf_counter() - start
    http_request_duration.labels(method=method, path=path).observe(duration)
    http_requests_total.labels(
        method=method, path=path, status=str(response.status_code)
    ).inc()
    if response.status_code >= 400:
        http_errors_total.labels(path=path, status=str(response.status_code)).inc()
    refresh_resource_metrics()
    return response


def metrics_endpoint():
    return generate_latest(REGISTRY)
