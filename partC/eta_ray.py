"""
eta_ray.py: AIOps Module 4, Assignment 5, Part C: a sticky canary for the ETA model on Ray Serve.

Architecture:
    client --HTTP--> Router (FastAPI ingress, sticky canary) --> EtaModelV1 (prod, 2 fixed replicas)
                                                             +-> EtaModelV2 (candidate, autoscaled 1..3)

Routing rules:
    1. The router hashes the customer_id with SHA-256 and a fixed salt. The result is a bucket in [0, 1).
    2. A customer with bucket < weight goes to the candidate (v2). All other customers go to prod (v1).
    3. The header "X-Beta: 1" sends that one request to the candidate.
    The hash is stable, so a customer always gets the same arm, also after a redeploy.
    A higher weight only adds customers to v2. A customer on v2 never goes back to v1.

CPU budget (the application must fit on 2 CPUs):
    2 x 0.25 (prod) + 3 x 0.25 (candidate at max) + 0.2 (router) = 1.45 CPUs.

Run:
    serve run partC.eta_ray:app_builder weight=0.2      (or call build(weight, extra_work_ms) from Python)
Env:
    ETA_V1, ETA_V2   paths to the model bundles (default data/eta_v1.joblib and data/eta_v2.joblib).

Citation: the deployment / router / build_app structure, the fractional num_cpus, the
health check and the extra_work_ms stand-in come from the lecture file ray_serve_app.py
(Lab2/DA3408, "Lecture 10 -- Ray_Serve_Exercise.ipynb"). The lecture router used
random() < weight per request. This file replaces it with a stable per-customer hash.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve
from ray.serve.handle import DeploymentHandle

# Configuration
# Keep the feature order the same as eta_data.FEATURES. The bundle also stores this order.
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]

# Do not change the salt during an experiment. A new salt gives all customers new buckets.
SALT = "eta-canary-v1v2"

PROD_NAME = "EtaModelV1"
CAND_NAME = "EtaModelV2"
ROUTER_NAME = "Router"

MODEL_CPUS = 0.25 # CPU reservation for one model replica.
ROUTER_CPUS = 0.2 # CPU reservation for the router replica.
PROD_REPLICAS = 2 # prod has a fixed number of replicas.

# Autoscaling for the candidate. See REPORT.pdf (C3) for the reasons for each value.
CAND_AUTOSCALING = dict(
    min_replicas=1, # Keep one warm replica. A cold start adds model load time.
    initial_replicas=1,
    max_replicas=3, # 3 x 0.25 CPU keeps the total below 2 CPUs.
    target_ongoing_requests=2, # One single-row predict takes ~1-3 ms. Two in flight per replica is a short queue.
    upscale_delay_s=3, # Add replicas fast when the queue grows.
    downscale_delay_s=30, # Remove replicas slowly. This prevents flapping.
    metrics_interval_s=1,
    look_back_period_s=5,
)

MAX_TOTAL_CPUS = PROD_REPLICAS * MODEL_CPUS + CAND_AUTOSCALING["max_replicas"] * MODEL_CPUS + ROUTER_CPUS

DEFAULT_V1 = "data/eta_v1.joblib"
DEFAULT_V2 = "data/eta_v2.joblib"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Pure routing functions (no Ray needed; easy to unit test)
def bucket(customer_id: str) -> float:
    """Return a stable number in [0, 1) for a customer id.

    Use SHA-256, not Python hash(). Python salts hash() for each process,
    so hash() gives a different value in each replica and after each restart.
    """
    digest = hashlib.sha256(f"{SALT}:{customer_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64

def choose_arm(customer_id: str, weight: float, beta: bool = False) -> str:
    """Return "v2" (candidate) or "v1" (prod) for one request.

    The beta flag forces the candidate. The rule "bucket < weight" is monotonic:
    a larger weight keeps all current v2 customers on v2.
    """
    if beta:
        return "v2"
    return "v2" if bucket(customer_id) < weight else "v1"

def resolve_model_path(path: str) -> str:
    """Return an absolute path to a model bundle.

    Ray workers can start in a different working directory. Thus resolve the path in the driver.
    Try the current directory first. Then try the repository root.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    for base in (Path.cwd(), REPO_ROOT):
        if (base / p).exists():
            return str((base / p).resolve())
    raise FileNotFoundError(f"model bundle not found: {path} (run 'python eta_data.py' first)")

# Request schema (same rules as Part A)
class Delivery(BaseModel):
    """One delivery. Each Field holds one business rule."""
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasible(self):
        # Reject a long route with many stops. The courier cannot do it in one trip.
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 and n_stops > 20")
        return self

class CanaryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    customer_id: str = Field(min_length=1, max_length=128)
    instances: List[Delivery] = Field(min_length=1, max_length=500)

# Model deployment
@serve.deployment(
    ray_actor_options={"num_cpus": MODEL_CPUS}, # Fractional CPU: many replicas share a few cores.
    max_ongoing_requests=16, # Back-pressure for each replica.
    health_check_period_s=5,
    logging_config={"log_level": "WARNING"},
)
class EtaModel:
    """Load one model bundle once per replica. Predict a batch of rows."""

    def __init__(self, path: str, extra_work_ms: float = 0.0):
        # extra_work_ms simulates a larger model. It is a blocking sleep, not CPU work.
        # Keep it at 0.0 for real work.
        self.extra_s = extra_work_ms / 1000.0
        bundle = joblib.load(path) # Load once per replica, never per request.
        self.model = bundle["model"]
        self.columns = bundle["feature_columns"]
        self.version = bundle["version"]
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[Dict]) -> Dict:
        X = pd.DataFrame(rows)[self.columns] # Use the column order from the bundle.
        y = self.model.predict(X)
        if self.extra_s:
            time.sleep(self.extra_s)
        return {"predictions": [round(float(v), 1) for v in y],
                "model_version": self.version, "served_by": self.replica}

    async def check_health(self):
        # Serve calls this method. An exception makes Serve restart the replica.
        if self.model is None:
            raise RuntimeError("model is not loaded")

# Router (FastAPI ingress)
api = FastAPI(title="SwiftParcel ETA sticky canary")

@api.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    """Return the same 422 body shape as Part A."""
    details = [{"field": ".".join(str(p) for p in e.get("loc", ()) if p != "body"), "msg": e.get("msg", "")}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details})

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": ROUTER_CPUS},
    logging_config={"log_level": "WARNING"},
)
@serve.ingress(api)
class Router:
    """Send each customer to a fixed arm. Return the model output plus the arm."""

    def __init__(self, v1: DeploymentHandle, v2: DeploymentHandle, weight: float):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.handles = {"v1": v1, "v2": v2}
        self.weight = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.weight}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: CanaryRequest, x_beta: Optional[str] = Header(default=None)):
        beta = (x_beta or "").strip() == "1"
        arm = choose_arm(req.customer_id, self.weight, beta=beta)
        rows = [d.model_dump() for d in req.instances]
        try:
            out = await self.handles[arm].remote(rows)
        except Exception:
            # Do not send internal details to the client.
            return JSONResponse(status_code=500, content={"error": "internal_error"})
        return {**out, "arm": arm, "customer_id": req.customer_id, "beta": beta}

# Application builder
def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Return the bound Serve application.

    prod (v1) has PROD_REPLICAS fixed replicas. The candidate (v2) uses CAND_AUTOSCALING.
    """
    v1_path = resolve_model_path(os.getenv("ETA_V1", DEFAULT_V1))
    v2_path = resolve_model_path(os.getenv("ETA_V2", DEFAULT_V2))
    prod = EtaModel.options(name=PROD_NAME, num_replicas=PROD_REPLICAS)
    cand = EtaModel.options(name=CAND_NAME, num_replicas="auto", autoscaling_config=dict(CAND_AUTOSCALING))
    return Router.options(name=ROUTER_NAME).bind(
        prod.bind(v1_path, extra_work_ms), cand.bind(v2_path, extra_work_ms), weight)

def app_builder(args: Dict[str, str]):
    """Return build() from "serve run" arguments, for example "weight=0.2 extra_work_ms=0"."""
    return build(weight=float(args.get("weight", 0.1)), extra_work_ms=float(args.get("extra_work_ms", 0.0)))
