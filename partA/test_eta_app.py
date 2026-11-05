"""
test_eta_app.py: Contract tests for the Part A ETA service (FastAPI TestClient).

Run from the repository root:
    python -m pytest -q partA/test_eta_app.py

Each test checks one rule of the contract. If you break that rule in eta_app.py, the test fails.
Idea from the Module 4 lecture: a table of invalid inputs, each one must give 422.
"""
import re
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent)) # Find eta_app from any folder.

import eta_app # noqa: E402
from fastapi.testclient import TestClient # noqa: E402

URL = "/v1/models/eta:predict"

VALID = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
         "traffic_index": 0.3, "rain": 0, "weight_kg": 8.0}

def row(**changes):
    """Make a copy of the valid row and change some fields."""
    return {**VALID, **changes}

def without(field):
    """Make a copy of the valid row without one field."""
    return {k: v for k, v in VALID.items() if k != field}

@pytest.fixture(scope="module")
def client():
    """Give one TestClient for all tests. The client does not raise server errors."""
    with TestClient(eta_app.app, raise_server_exceptions=False) as c:
        yield c

def metric_value(text, name, labels=""):
    """Read one sample value from Prometheus text. Give 0.0 if the sample is not there."""
    pattern = rf"^{re.escape(name)}{re.escape(labels)} ([0-9.e+-]+)$"
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else 0.0

# health and valid requests
def test_healthz_reports_status_and_version(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}

def test_valid_request_returns_rounded_prediction(client):
    r = client.post(URL, json={"instances": [VALID]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert body["model_version"] == eta_app.MODEL_VERSION
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p > 0
    assert round(p, 1) == p # One decimal only.

def test_batch_order_is_preserved(client):
    rows = [row(distance_km=d) for d in (2.0, 80.0, 15.0, 140.0, 0.5)]
    batch = client.post(URL, json={"instances": rows}).json()["predictions"]
    single = [client.post(URL, json={"instances": [r]}).json()["predictions"][0] for r in rows]
    assert batch == single # Same answer per row, same order.
    rev = client.post(URL, json={"instances": rows[::-1]}).json()["predictions"]
    assert rev == batch[::-1]
    assert len(set(batch)) == len(batch) # Rows differ, so the order matters.

def test_extra_fields_are_ignored(client):
    r = client.post(URL, json={"instances": [row(eta_minutes=99.9)], "customer_id": "c-1"})
    assert r.status_code == 200

# invalid input (422)
@pytest.mark.parametrize("instance, field", [
    (row(hour=24), "instances.0.hour"), # Out of range (high).
    (row(distance_km=0.05), "instances.0.distance_km"), # Out of range (low).
    (row(weight_kg=0), "instances.0.weight_kg"), # Must be > 0.
    (row(weight_kg=1000.01), "instances.0.weight_kg"), # Must be <= 1000.
    (row(rain=2), "instances.0.rain"), # Must be 0 or 1.
    (row(traffic_index=1.5), "instances.0.traffic_index"),
    (row(weekday=7), "instances.0.weekday"),
    (row(n_stops=0), "instances.0.n_stops"),
    (row(n_stops=61), "instances.0.n_stops"),
    (row(n_stops=3.5), "instances.0.n_stops"), # Not an integer.
    (row(rain=True), "instances.0.rain"), # Boolean, not 0 or 1.
    (row(distance_km="far"), "instances.0.distance_km"), # Wrong type (string).
    (without("hour"), "instances.0.hour"), # Missing field.
])
def test_invalid_instance_gives_422_with_field(client, instance, field):
    r = client.post(URL, json={"instances": [instance]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert field in [d["field"] for d in body["details"]]
    assert all(set(d) == {"field", "msg"} for d in body["details"])

@pytest.mark.parametrize("payload", [
    {"rows": [VALID]}, # Wrong top-level key.
    {"instances": VALID}, # Object, not a list.
    {}, # Empty body.
])
def test_bad_request_shape_gives_422(client, payload):
    r = client.post(URL, json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"

def test_malformed_json_gives_422(client):
    r = client.post(URL, content=b'{"instances": [', headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"

# cross-field rule
def test_infeasible_delivery_is_rejected(client):
    r = client.post(URL, json={"instances": [VALID, row(distance_km=150.1, n_stops=21)]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert any("infeasible" in d["msg"] for d in body["details"])
    assert any(d["field"].startswith("instances.1") for d in body["details"])

@pytest.mark.parametrize("instance", [row(distance_km=150, n_stops=60),
                                      row(distance_km=500, n_stops=20)])
def test_cross_field_boundary_is_feasible(client, instance):
    assert client.post(URL, json={"instances": [instance]}).status_code == 200

# batch-size limits
def test_empty_batch_is_rejected(client):
    r = client.post(URL, json={"instances": []})
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "instances"

def test_batch_of_500_is_accepted(client):
    r = client.post(URL, json={"instances": [VALID] * 500})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 500

def test_batch_of_501_is_rejected(client):
    r = client.post(URL, json={"instances": [VALID] * 501})
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "instances"

# request ID
def test_request_id_is_echoed(client):
    r = client.post(URL, json={"instances": [VALID]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"

def test_long_request_id_is_echoed(client):
    rid = "trace-" + "x" * 300
    r = client.post(URL, json={"instances": [VALID]}, headers={"X-Request-ID": rid})
    assert r.headers["X-Request-ID"] == rid == r.json()["request_id"]

def test_request_id_is_generated_as_uuid(client):
    r = client.post(URL, json={"instances": [VALID]})
    rid = r.headers["X-Request-ID"]
    assert str(uuid.UUID(rid)) == rid
    assert r.json()["request_id"] == rid

def test_request_id_on_error_echo_and_generate(client):
    echoed = client.post(URL, json={"instances": []}, headers={"X-Request-ID": "err-42"})
    assert echoed.status_code == 422
    assert echoed.headers["X-Request-ID"] == "err-42" == echoed.json()["request_id"]
    generated = client.post(URL, json={"instances": [row(hour=99)]})
    rid = generated.headers["X-Request-ID"]
    assert uuid.UUID(rid) and generated.json()["request_id"] == rid

def test_request_id_header_on_other_routes(client):
    assert "X-Request-ID" in client.get("/healthz").headers
    assert "X-Request-ID" in client.get("/no-such-path").headers

# metrics
def test_metrics_count_requests_and_latency(client):
    label = f'{{path="{URL}",status="200"}}'
    before = metric_value(client.get("/metrics").text, "eta_requests_total", label)
    count_before = metric_value(client.get("/metrics").text, "eta_request_latency_seconds_count")
    client.post(URL, json={"instances": [VALID]})
    client.post(URL, json={"instances": [VALID]})
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert metric_value(text, "eta_requests_total", label) == before + 2
    assert metric_value(text, "eta_request_latency_seconds_count") == count_before + 2
    assert len(re.findall(r'^eta_request_latency_seconds_bucket\{le="', text, re.M)) >= 5
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert "eta_request_latency_seconds_sum " in text
    assert "# TYPE eta_requests_total counter" in text
    assert "# TYPE eta_request_latency_seconds histogram" in text

def test_metrics_count_errors_by_status(client):
    label = f'{{path="{URL}",status="422"}}'
    before = metric_value(client.get("/metrics").text, "eta_requests_total", label)
    client.post(URL, json={"instances": []})
    assert metric_value(client.get("/metrics").text, "eta_requests_total", label) == before + 1

def test_metrics_endpoint_is_not_counted(client):
    for _ in range(3):
        text = client.get("/metrics").text
    assert 'path="/metrics"' not in text
    count = metric_value(text, "eta_request_latency_seconds_count")
    assert metric_value(client.get("/metrics").text, "eta_request_latency_seconds_count") == count

# server errors and loading
def test_inference_failure_gives_generic_500(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail: db password=hunter2")

    monkeypatch.setattr(eta_app, "model", Broken())
    r = client.post(URL, json={"instances": [VALID]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert "secret" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"

def test_model_is_not_loaded_per_request(client, monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError("joblib.load was called during a request")

    monkeypatch.setattr(eta_app.joblib, "load", fail_load)
    assert client.post(URL, json={"instances": [VALID]}).status_code == 200
