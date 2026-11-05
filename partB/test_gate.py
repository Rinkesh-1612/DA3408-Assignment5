"""
test_gate.py: Tests of the pure promotion rule decide() in eta_mlflow.py (Part B2).

The tests use synthetic numbers only. They do not start MLflow and do not read files.
The boundary cases use clear margins (1e-6). Thus floating-point rounding does not change the result.
Run: python -m pytest -q partB/test_gate.py
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eta_mlflow import decide # noqa: E402

# Prod metrics. The default thresholds: gain 0.03, p95 10 ms, slice regression 5 % (relative).
PROD = {"mae": 10.0, "mae_rush": 20.0, "mae_rain": 16.0, "p95_ms": 2.0}
EPS = 1e-6

def cand(**changes):
    """Return a candidate that passes all criteria. Apply the changes to it."""
    base = {"mae": 6.0, "mae_rush": 12.0, "mae_rain": 10.0, "p95_ms": 5.0}
    return {**base, **changes}

def failed(result):
    """Return the names of the criteria that did not pass."""
    return {k for k, v in result["criteria"].items() if not v["passed"]}

def test_all_criteria_pass_promotes():
    r = decide(PROD, cand())
    assert r["promote"] is True
    assert failed(r) == set()
    assert set(r["criteria"]) == {"mae_gain", "p95_latency", "rush_slice", "rain_slice"}

@pytest.mark.parametrize("mae, promote", [
    (10.0 * (1 - 0.03) - EPS, True), # gain just above 3 %
    (10.0 * (1 - 0.03) + EPS, False), # gain just below 3 %
])
def test_mae_gain_boundary(mae, promote):
    r = decide(PROD, cand(mae=mae))
    assert r["promote"] is promote
    assert ("mae_gain" in failed(r)) is (not promote)

def test_no_gain_blocks():
    r = decide(PROD, cand(mae=10.0))
    assert r["promote"] is False and failed(r) == {"mae_gain"}

@pytest.mark.parametrize("p95, promote", [(10.0 - EPS, True), (10.0, True), (10.0 + EPS, False)])
def test_p95_boundary(p95, promote):
    r = decide(PROD, cand(p95_ms=p95))
    assert r["promote"] is promote
    assert ("p95_latency" in failed(r)) is (not promote)

@pytest.mark.parametrize("key, crit, prod_value", [("mae_rush", "rush_slice", 20.0),
                                                   ("mae_rain", "rain_slice", 16.0)])
@pytest.mark.parametrize("delta, promote", [(-EPS, True), (+EPS, False)])
def test_slice_boundary(key, crit, prod_value, delta, promote):
    limit = prod_value * 1.05 # 5 % relative regression is the limit
    r = decide(PROD, cand(**{key: limit + delta}))
    assert r["promote"] is promote
    assert failed(r) == (set() if promote else {crit})

def test_slice_regression_blocks_even_with_big_overall_gain():
    # The overall MAE is much better, but the rain slice is 50 % worse. Do not promote.
    r = decide(PROD, cand(mae=1.0, mae_rain=24.0))
    assert r["promote"] is False and failed(r) == {"rain_slice"}

def test_every_criterion_fails_together():
    r = decide(PROD, {"mae": 11.0, "mae_rush": 30.0, "mae_rain": 30.0, "p95_ms": 50.0})
    assert r["promote"] is False
    assert failed(r) == {"mae_gain", "p95_latency", "rush_slice", "rain_slice"}

def test_custom_thresholds():
    c = cand() # gain is 40 %
    assert decide(PROD, c, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, c, min_rel_gain=0.40 - EPS)["promote"] is True
    assert decide(PROD, c, max_p95_ms=4.0)["promote"] is False
    # Allow a 30 % slice regression: a candidate 25 % worse on rush now passes.
    worse_rush = cand(mae_rush=25.0)
    assert decide(PROD, worse_rush)["promote"] is False
    assert decide(PROD, worse_rush, max_slice_regression=0.30)["promote"] is True

def test_output_shape_and_values():
    r = decide(PROD, cand())
    g = r["criteria"]["mae_gain"]
    assert g["value"] == pytest.approx(0.4) and g["threshold"] == pytest.approx(0.03)
    assert r["criteria"]["rush_slice"]["threshold"] == pytest.approx(21.0)
    for v in r["criteria"].values():
        assert isinstance(v["passed"], bool)

def test_decide_is_pure():
    p, c = copy.deepcopy(PROD), cand()
    p0, c0 = copy.deepcopy(p), copy.deepcopy(c)
    first, second = decide(p, c), decide(p, c)
    assert first == second # same input -> same output
    assert p == p0 and c == c0 # inputs are not changed
