"""eta_app.py: the delivery time (ETA) service for Assignment 5, Part A.

Start the service:
    ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000

Routes:
    POST /v1/models/eta:predict gives the delivery time in minutes for 1 to 500 deliveries.
    GET /healthz shows that the service runs and which model version it uses.
    GET /metrics shows the Prometheus metrics.

Rules from the Module 4 lecture (rest_app.py):
Load the model one time, at start-up.
Check the input at the edge. Bad input gives 422. A server fault gives a generic 500.
Put the model version in each answer.
Use a plain "def" for the predict route. FastAPI then runs it in a thread pool.
"""
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram,
                               disable_created_metrics, generate_latest)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Settings
DEFAULT_MODEL_PATH = "data/eta_v1.joblib"
REQUEST_ID_HEADER = "X-Request-ID"
PREDICT_PATH = "/v1/models/eta:predict"
METRICS_PATH = "/metrics"
MAX_BATCH = 500

# A delivery with a long distance and many stops is not possible.
INFEASIBLE_DISTANCE_KM = 150
INFEASIBLE_STOPS = 20

# Latency buckets in seconds.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5)

def find_model_file(name):
    """Look for the model file in the current folder. Then look in the repository root."""
    path = Path(name)
    if path.is_absolute() or path.exists():
        return path
    in_root = Path(__file__).resolve().parent.parent / name
    return in_root if in_root.exists() else path

# Load the model one time.
bundle = joblib.load(find_model_file(os.getenv("ETA_MODEL", DEFAULT_MODEL_PATH)))
model = bundle["model"]
FEATURE_COLUMNS = list(bundle["feature_columns"])
MODEL_VERSION = str(bundle["version"])

# Input rules
class Delivery(BaseModel):
    """One delivery. Each range is a business rule. Unknown fields are ignored."""
    model_config = ConfigDict(extra="ignore")

    distance_km: float = Field(ge=0.1, le=500, allow_inf_nan=False)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1, allow_inf_nan=False)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000, allow_inf_nan=False)

    @field_validator("n_stops", "hour", "weekday", "rain", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        """Do not accept true or false. Pydantic would change them to 1 or 0."""
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def reject_infeasible(self):
        """Do not accept a long delivery with many stops."""
        if self.distance_km > INFEASIBLE_DISTANCE_KM and self.n_stops > INFEASIBLE_STOPS:
            raise ValueError(f"infeasible delivery: distance_km > {INFEASIBLE_DISTANCE_KM} "
                             f"and n_stops > {INFEASIBLE_STOPS}")
        return self

class PredictRequest(BaseModel):
    """The request body. It has 1 to 500 deliveries."""
    instances: list[Delivery] = Field(min_length=1, max_length=MAX_BATCH)

# Metrics
disable_created_metrics()
REGISTRY = CollectorRegistry()
REQUESTS = Counter("eta_requests", "Requests by path and status code.",
                   ["path", "status"], registry=REGISTRY)
LATENCY = Histogram("eta_request_latency_seconds", "Request time in seconds.",
                    buckets=LATENCY_BUCKETS, registry=REGISTRY)

# Application
app = FastAPI(title="SwiftParcel ETA service")

def internal_error(request_id):
    """Make a generic 500 answer. Do not show internal details."""
    return JSONResponse(status_code=500,
                        content={"error": "internal_error", "request_id": request_id})

@app.middleware("http")
async def add_request_id_and_count(request: Request, call_next):
    """Set the request ID. Time and count the request. Put the ID in the answer header."""
    client_id = request.headers.get(REQUEST_ID_HEADER, "")
    request.state.request_id = client_id if client_id.strip() else str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        response = internal_error(request.state.request_id)
    seconds = time.perf_counter() - start
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    # Do not count the calls to /metrics.
    if request.url.path != METRICS_PATH:
        # Use the route template as the path label.
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(path=path, status=str(response.status_code)).inc()
        LATENCY.observe(seconds)
    return response

@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, error: RequestValidationError):
    """Give 422 with the error name, the list of bad fields and the request ID."""
    details = []
    for item in error.errors():
        names = [str(part) for part in item["loc"] if part != "body"]
        details.append({"field": ".".join(names) or "body", "msg": str(item["msg"])})
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,
                                                  "request_id": request.state.request_id})

# Routes
@app.post(PREDICT_PATH)
def predict(body: PredictRequest, request: Request):
    """Give the delivery time in minutes for each delivery. Keep the input order."""
    try:
        rows = pd.DataFrame([d.model_dump() for d in body.instances])[FEATURE_COLUMNS]
        minutes = [round(float(m), 1) for m in model.predict(rows)]
    except Exception:
        return internal_error(request.state.request_id)
    return {"predictions": minutes, "model_version": MODEL_VERSION,
            "request_id": request.state.request_id}

@app.get("/healthz")
def healthz():
    """Show that the service runs and which model version it uses."""
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get(METRICS_PATH)
def metrics():
    """Give the metrics in the Prometheus text format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
