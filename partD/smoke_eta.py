"""
smoke_eta.py: Release gate for the SwiftParcel ETA service. AIOps Module 4, Assignment 5, Part D.

Run this script after a deploy and before the new version gets real traffic.
Exit code 0 means "release allowed". Exit code 1 means "stop the release and roll back".

Usage:
    python partD/smoke_eta.py --url http://localhost:8010 --expect-version v1 [--max-p95-ms 80]
    python partD/smoke_eta.py --url http://localhost:8100 --expect-version v2 --beta     (Ray, Part C)

--url accepts the base URL or the full predict URL (.../v1/models/eta:predict).
Each request also sends "customer_id". Part A ignores it. The Ray router (Part C) needs it.

Checks (8):
    1. health        GET /healthz gives 200 and status "ok".
    2. version       A valid request gives 200 and the expected model_version.
    3. contract      A batch gives one finite prediction per row, rounded to one decimal.
    4. sanity-dist   A longer distance never gives a shorter ETA (other features fixed).
    5. sanity-traffic Heavier traffic never gives a shorter ETA.
    6. sanity-range  ETAs for realistic deliveries are in the range 1 to 600 minutes.
    7. invalid-422   Out-of-range input and an infeasible delivery give HTTP 422.
    8. latency       p95 of 30 single-row requests (after warm-up) is at most --max-p95-ms.
Idea from the Module 4 lecture (smoke_test.py): health, version, latency bound, 422 check, exit code.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable, List, Tuple

import requests

PREDICT_ROUTE = "/v1/models/eta:predict"
TIMEOUT_S = 5.0
CUSTOMER_ID = "smoke-test-0001"

BASE_ROW = {"distance_km": 25.0, "n_stops": 5, "hour": 11, "weekday": 2,
            "traffic_index": 0.4, "rain": 0, "weight_kg": 10.0}

# HTTP helpers
class Client:
    """Small HTTP client. It keeps one session and adds the optional X-Beta header."""

    def __init__(self, url: str, beta: bool):
        url = url.rstrip("/")
        self.base = url[: -len(PREDICT_ROUTE)] if url.endswith(PREDICT_ROUTE) else url
        self.predict_url = self.base + PREDICT_ROUTE
        self.session = requests.Session()
        self.headers = {"X-Beta": "1"} if beta else {}

    def predict(self, instances: List[dict]) -> requests.Response:
        """Send one predict request. Do not retry: a smoke test must see each failure."""
        body = {"customer_id": CUSTOMER_ID, "instances": instances}
        return self.session.post(self.predict_url, json=body, headers=self.headers, timeout=TIMEOUT_S)

    def etas(self, instances: List[dict]) -> List[float]:
        """Give the predictions. Raise an error if the status is not 200."""
        r = self.predict(instances)
        r.raise_for_status()
        return r.json()["predictions"]

def row(**changes) -> dict:
    """Make a copy of the base row and change some fields."""
    return {**BASE_ROW, **changes}

def non_decreasing(values: List[float]) -> bool:
    """Tell if each value is at least as large as the value before it."""
    return all(b >= a for a, b in zip(values, values[1:]))

# checks
# Each check returns (passed, evidence). An exception in a check counts as a failure.
def check_health(c: Client, a) -> Tuple[bool, str]:
    r = c.session.get(c.base + "/healthz", timeout=TIMEOUT_S)
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    return ok, f"status={r.status_code} body={r.text[:120]}"

def check_version(c: Client, a) -> Tuple[bool, str]:
    r = c.predict([BASE_ROW])
    got = r.json().get("model_version") if r.status_code == 200 else None
    return r.status_code == 200 and got == a.expect_version, f"status={r.status_code} model_version={got!r}"

def check_contract(c: Client, a) -> Tuple[bool, str]:
    rows = [row(distance_km=d) for d in (3.0, 45.0, 12.0)]
    preds = c.etas(rows)
    finite = all(isinstance(p, (int, float)) and math.isfinite(p) for p in preds)
    one_decimal = all(round(p, 1) == p for p in preds) if finite else False
    ok = len(preds) == len(rows) and finite and one_decimal
    return ok, f"rows={len(rows)} predictions={preds}"

def check_distance(c: Client, a) -> Tuple[bool, str]:
    dists = [2.0, 10.0, 30.0, 60.0, 120.0]
    preds = c.etas([row(distance_km=d) for d in dists])
    return non_decreasing(preds), f"distance_km={dists} -> eta={preds}"

def check_traffic(c: Client, a) -> Tuple[bool, str]:
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    preds = c.etas([row(distance_km=40.0, hour=8, traffic_index=t) for t in levels])
    return non_decreasing(preds), f"traffic_index={levels} -> eta={preds}"

def check_range(c: Client, a) -> Tuple[bool, str]:
    rows = [row(distance_km=d, n_stops=s, hour=h, rain=r)
            for d, s, h, r in [(0.5, 1, 3, 0), (5, 2, 8, 1), (20, 10, 17, 1), (80, 15, 12, 0), (150, 20, 18, 1)]]
    preds = c.etas(rows)
    return all(1.0 <= p <= 600.0 for p in preds), f"eta={preds} (allowed 1..600 min)"

def check_invalid(c: Client, a) -> Tuple[bool, str]:
    cases = {"hour=25": [row(hour=25)], "distance_km=-5": [row(distance_km=-5)],
             "infeasible 200km/30 stops": [row(distance_km=200, n_stops=30)], "empty batch": []}
    codes = {name: c.predict(inst).status_code for name, inst in cases.items()}
    return all(code == 422 for code in codes.values()), f"status codes={codes}"

def check_latency(c: Client, a) -> Tuple[bool, str]:
    for _ in range(5): # Warm-up. Discard these timings.
        c.predict([BASE_ROW])
    lat = []
    for i in range(30):
        t0 = time.perf_counter()
        r = c.predict([row(distance_km=5.0 + i)])
        lat.append((time.perf_counter() - t0) * 1000.0)
        if r.status_code != 200:
            return False, f"request {i} gave status {r.status_code}"
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return p95 <= a.max_p95_ms, f"p50={lat[len(lat) // 2]:.1f} ms p95={p95:.1f} ms (limit {a.max_p95_ms:g} ms)"

CHECKS: List[Tuple[str, Callable]] = [
    ("health", check_health), ("version", check_version), ("contract", check_contract),
    ("sanity-distance", check_distance), ("sanity-traffic", check_traffic), ("sanity-range", check_range),
    ("invalid-422", check_invalid), ("latency-p95", check_latency)]

# main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Smoke test and release gate for the ETA service.")
    ap.add_argument("--url", required=True, help="base URL or full predict URL")
    ap.add_argument("--expect-version", required=True, help="model_version the service must report")
    ap.add_argument("--max-p95-ms", type=float, default=80.0, help="latency limit for check 8")
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray router: force the candidate)")
    a = ap.parse_args(argv)

    c = Client(a.url, a.beta)
    print(f"Smoke test: {c.predict_url}  expect-version={a.expect_version}  max-p95-ms={a.max_p95_ms:g}")
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, evidence = fn(c, a)
        except Exception as e: # A crash in a check is a failed check.
            ok, evidence = False, f"error: {type(e).__name__}: {e}"
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} {evidence}")
    verdict = "RELEASE ALLOWED" if failed == 0 else f"RELEASE BLOCKED ({failed} check(s) failed)"
    print(f"Result: {len(CHECKS) - failed}/{len(CHECKS)} checks passed -> {verdict}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
