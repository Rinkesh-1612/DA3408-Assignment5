"""
eta_mlflow.py: SwiftParcel ETA, Part B: MLflow registry, promotion gate and shadow comparison.

This module does three tasks:
  1. register()        Wrap each joblib bundle in a pyfunc. Log it with a signature and an input example.
                       Register both as "eta-model". Set the aliases prod -> v1 and candidate -> v2.
  2. promotion_gate()  Load both aliases. Evaluate them on fresh data. Measure single-row p95.
                       Call decide(). decide() is a pure function.
  3. shadow()          Send the same 200 rows to two "mlflow models serve" servers. Compare the answers.

Command line (run from the repository root):
  python partB/eta_mlflow.py register
  python partB/eta_mlflow.py gate   [--out partB/gate_output.txt]
  python partB/eta_mlflow.py shadow --prod-url http://localhost:5101/invocations \
                                    --cand-url http://localhost:5102/invocations [--out partB/shadow_output.txt]

Environment:
  MLFLOW_TRACKING_URI  Tracking store. Default: sqlite:///mlflow.db. The registry needs a database store.
  ETA_ARTIFACT_ROOT    Artifact location of the experiment. Default: ./mlartifacts (made absolute).
  ETA_DATA_DIR         Folder with the data and the bundles. Default: data.

Citation: the pyfunc wrapper, the float64 signature and the alias steps follow the idea of
mlflow_register.py and the Module 4 lecture "MLflow Serving" (AIOps, DA3408). The code is new.
"""
import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import joblib
import numpy as np
import pandas as pd
import requests

# Make "eta_data" importable. It is in the repository root, one level above this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import eta_data # noqa: E402  (import after the sys.path change)
import mlflow # noqa: E402
import mlflow.pyfunc # noqa: E402
from mlflow import MlflowClient # noqa: E402
from mlflow.models import infer_signature # noqa: E402

# Configuration
MODEL_NAME = "eta-model"
EXPERIMENT = "eta-serving"
FEATURES = list(eta_data.FEATURES)
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
VERSIONS = {"prod": "v1", "candidate": "v2"} # alias -> bundle version
FRESH_N, FRESH_SEED = 2000, 99 # fresh data: no model saw it in training
P95_CALLS, WARMUP_CALLS = 200, 20

def tracking_uri():
    """Return the tracking URI. Read it from MLFLOW_TRACKING_URI."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

def data_dir():
    """Return the data folder.

    Use ETA_DATA_DIR relative to the current folder. If that folder does not exist,
    use the "data" folder in the repository root.
    """
    d = os.getenv("ETA_DATA_DIR", "data")
    if os.path.isdir(d):
        return os.path.abspath(d)
    return os.path.join(REPO_ROOT, "data")

def bundle_path(ver):
    """Return the path of the joblib bundle for one model version (v1 or v2)."""
    return os.path.join(data_dir(), f"eta_{ver}.joblib")

def setup_mlflow():
    """Set the tracking URI. Make the experiment with a known artifact location.

    The artifact location must be the same path in all processes. In Docker, the register job
    and the two serve containers share one volume (/state). Thus they find the same artifacts.
    """
    mlflow.set_tracking_uri(tracking_uri())
    root = os.path.abspath(os.getenv("ETA_ARTIFACT_ROOT", "mlartifacts"))
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        os.makedirs(root, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=root)
    mlflow.set_experiment(EXPERIMENT)

def machine_info():
    """Return a short description of this machine. The report must state the machine."""
    mem = "n/a"
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
            mem = f"{kb / 1024 / 1024:.1f} GiB"
    except Exception:
        pass
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "usable_cpus": cpus, "memory": mem,
            "mlflow": mlflow.__version__}

# The pyfunc wrapper
class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wrap one ETA bundle. Return eta_minutes (one decimal) and model_version.

    The bundle is a dict {model, feature_columns, version}. Use the stored column order.
    Do not guess the order of the columns.
    """

    def load_context(self, context):
        """Load the bundle one time, when the model loads."""
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), b["version"]

    def predict(self, context, model_input, params=None):
        """Predict the delivery time in minutes for each row."""
        X = pd.DataFrame(model_input)[self.cols].astype("float64")
        y = np.round(self.model.predict(X).astype("float64"), 1)
        return pd.DataFrame({"eta_minutes": y, "model_version": [self.version] * len(y)})

def pinned_requirements():
    """Return exact versions of the packages that the model needs at serve time."""
    pkgs = ["mlflow", "scikit-learn", "pandas", "numpy", "scipy", "joblib"]
    return [f"{p}=={pkg_version(p)}" for p in pkgs]

# Register (B1)
def register():
    """Log v1 and v2 as pyfunc models. Register them. Set the aliases prod and candidate.

    Each call makes two new registry versions. The aliases then point to the new pair.
    Return the registry table as a list of dicts (version, aliases, bundle_version).
    """
    setup_mlflow()
    client = MlflowClient()
    fresh = pd.read_csv(os.path.join(data_dir(), "eta_fresh.csv"))
    # Use float64 inputs. JSON integers and JSON floats then both agree with the signature.
    X = fresh[FEATURES].astype("float64")
    out_example = pd.DataFrame({"eta_minutes": [42.5], "model_version": ["v1"]})
    signature = infer_signature(X.head(5), out_example)

    registered = {}
    for alias, ver in VERSIONS.items():
        path = bundle_path(ver)
        b = joblib.load(path)
        pred = b["model"].predict(fresh[b["feature_columns"]])
        mae = float(np.mean(np.abs(fresh["eta_minutes"].to_numpy() - pred)))
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_param("algorithm", type(b["model"]).__name__)
            mlflow.log_param("bundle_version", ver)
            mlflow.log_metric("fresh_mae", mae)
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": path},
                signature=signature, input_example=X.head(2),
                pip_requirements=pinned_requirements(),
                registered_model_name=MODEL_NAME)
        mv = info.registered_model_version
        client.set_model_version_tag(MODEL_NAME, str(mv), "bundle_version", ver)
        client.set_registered_model_alias(MODEL_NAME, alias, str(mv))
        registered[alias] = mv
        print(f"logged {ver} ({type(b['model']).__name__}) fresh MAE={mae:.2f} -> {MODEL_NAME} version {mv}")

    table = registry_table(client)
    print(format_registry_table(table))
    return table

def registry_table(client=None):
    """Return all versions of eta-model with their aliases and bundle versions."""
    client = client or MlflowClient()
    rows = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        full = client.get_model_version(MODEL_NAME, mv.version)
        rows.append({"version": int(full.version), "aliases": ", ".join(sorted(full.aliases)) or "-",
                     "bundle_version": full.tags.get("bundle_version", "-")})
    return sorted(rows, key=lambda r: r["version"])

def format_registry_table(rows):
    """Return the registry table as Markdown text."""
    lines = ["| version | alias | bundle |", "|---|---|---|"]
    lines += [f"| {r['version']} | {r['aliases']} | {r['bundle_version']} |" for r in rows]
    return "\n".join(lines)

# Promotion gate (B2)
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Decide if the candidate can replace prod. This function is pure: no I/O, no state.

    Inputs: two dicts with the keys mae, mae_rush, mae_rain and p95_ms.
    Promote only if ALL criteria pass:
      mae_gain      (prod.mae - cand.mae) / prod.mae >= min_rel_gain
      p95_latency   cand.p95_ms <= max_p95_ms
      rush_slice    cand.mae_rush <= prod.mae_rush * (1 + max_slice_regression)
      rain_slice    cand.mae_rain <= prod.mae_rain * (1 + max_slice_regression)
    Decision: max_slice_regression is RELATIVE (0.05 = 5 % worse than prod). A relative limit
    has the same meaning for slices with small and large errors.
    The values in the output are rounded for display. The comparisons use the exact values.
    """
    def crit(value, threshold, passed, rule):
        return {"value": round(float(value), 4), "threshold": round(float(threshold), 4),
                "rule": rule, "passed": bool(passed)}

    gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    rush_limit = prod["mae_rush"] * (1 + max_slice_regression)
    rain_limit = prod["mae_rain"] * (1 + max_slice_regression)
    criteria = {
        "mae_gain": crit(gain, min_rel_gain, gain >= min_rel_gain, "relative MAE gain >= threshold"),
        "p95_latency": crit(cand["p95_ms"], max_p95_ms, cand["p95_ms"] <= max_p95_ms,
                            "candidate p95_ms <= threshold"),
        "rush_slice": crit(cand["mae_rush"], rush_limit, cand["mae_rush"] <= rush_limit,
                           "candidate mae_rush <= prod mae_rush * (1 + max_slice_regression)"),
        "rain_slice": crit(cand["mae_rain"], rain_limit, cand["mae_rain"] <= rain_limit,
                           "candidate mae_rain <= prod mae_rain * (1 + max_slice_regression)"),
    }
    return {"criteria": criteria, "promote": all(c["passed"] for c in criteria.values())}

def evaluate(model, df):
    """Return the gate metrics of one loaded pyfunc model on the data frame df."""
    X = df[FEATURES].astype("float64")
    err = np.abs(model.predict(X)["eta_minutes"].to_numpy() - df["eta_minutes"].to_numpy())
    rush = df["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (df["rain"] == 1).to_numpy()
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()),
            "mae_rain": float(err[rain].mean()), "p95_ms": single_row_p95(model, X),
            "n_rows": int(len(df)), "n_rush": int(rush.sum()), "n_rain": int(rain.sum())}

def single_row_p95(model, X, n=P95_CALLS, warmup=WARMUP_CALLS):
    """Measure the p95 latency (ms) of n single-row predictions. Do warm-up calls first."""
    rows = [X.iloc[[i]] for i in range(n)]
    for i in range(warmup):
        model.predict(rows[i % n])
    lat = []
    for r in rows:
        t0 = time.perf_counter()
        model.predict(r)
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(lat, 95))

def promotion_gate(**thresholds):
    """Load both aliases. Evaluate them on fresh data. Return the metrics and the decision."""
    setup_mlflow()
    fresh = eta_data.make_data(FRESH_N, seed=FRESH_SEED)
    metrics = {}
    for alias in ("prod", "candidate"):
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        metrics[alias] = evaluate(model, fresh)
    decision = decide(metrics["prod"], metrics["candidate"], **thresholds)
    return {"thresholds": {**_default_thresholds(), **thresholds},
            "prod": metrics["prod"], "candidate": metrics["candidate"], **decision}

def _default_thresholds():
    """Return the default thresholds of decide()."""
    return {"min_rel_gain": 0.03, "max_p95_ms": 10.0, "max_slice_regression": 0.05}

def format_gate(result, title):
    """Return the gate result as readable text."""
    p, c = result["prod"], result["candidate"]
    lines = [f"== {title} ==", f"thresholds: {result['thresholds']}",
             f"{'metric':<10}{'prod':>10}{'candidate':>12}"]
    for k in ("mae", "mae_rush", "mae_rain", "p95_ms"):
        lines.append(f"{k:<10}{p[k]:>10.3f}{c[k]:>12.3f}")
    lines.append(f"rows: {p['n_rows']} (rush {p['n_rush']}, rain {p['n_rain']})")
    lines.append(f"{'criterion':<14}{'value':>10}{'threshold':>11}  passed")
    for k, v in result["criteria"].items():
        lines.append(f"{k:<14}{v['value']:>10.4f}{v['threshold']:>11.4f}  {v['passed']}")
    lines.append(f"PROMOTE: {result['promote']}")
    return "\n".join(lines)

# Shadow comparison (B3)
def wait_ready(invocations_url, timeout=120):
    """Poll the /ping route of a server until it answers 200. Raise an error after the timeout."""
    ping = invocations_url.rsplit("/", 1)[0] + "/ping"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(ping, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{ping} is not ready after {timeout} s")

def _call(sess, url, row):
    """Send one row. Return (eta_minutes or None, latency ms, model_version or None)."""
    t0 = time.perf_counter()
    try:
        r = sess.post(url, json={"dataframe_records": [row]}, timeout=10)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return None, ms, None
        pred = r.json()["predictions"][0]
        return float(pred["eta_minutes"]), ms, pred.get("model_version")
    except Exception:
        return None, (time.perf_counter() - t0) * 1000, None

def _pct(values):
    """Return p50, p95 and p99 of a list, in ms."""
    if not values:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {f"p{q}_ms": round(float(np.percentile(values, q)), 2) for q in (50, 95, 99)}

def shadow(prod_url, cand_url, n=200, csv=None, diff_minutes=10.0):
    """Send the same n rows (one row per request) to both servers. Compare the answers.

    Alternate the order of the two calls on each row. Thus no server always gets a warm cache.
    """
    csv = csv or os.path.join(data_dir(), "eta_fresh.csv")
    rows = pd.read_csv(csv)[FEATURES].head(n).to_dict(orient="records")
    for url in (prod_url, cand_url):
        wait_ready(url)
    s_prod, s_cand = requests.Session(), requests.Session()
    for row in rows[:10]: # warm-up: not in the results
        _call(s_prod, prod_url, row)
        _call(s_cand, cand_url, row)

    res = {"prod": [], "candidate": []}
    for i, row in enumerate(rows):
        order = [("prod", s_prod, prod_url), ("candidate", s_cand, cand_url)]
        for name, sess, url in (order if i % 2 == 0 else order[::-1]):
            res[name].append(_call(sess, url, row))

    pairs = [(a[0], b[0]) for a, b in zip(res["prod"], res["candidate"])
             if a[0] is not None and b[0] is not None]
    diff = np.abs(np.array([a - b for a, b in pairs])) if pairs else np.array([])
    out = {"n_requests": len(rows), "n_compared": len(pairs),
           "mean_abs_diff_min": round(float(diff.mean()), 3) if len(diff) else None,
           "pct_diff_gt_10min": round(float((diff > diff_minutes).mean() * 100), 2) if len(diff) else None,
           "max_abs_diff_min": round(float(diff.max()), 2) if len(diff) else None}
    for name in ("prod", "candidate"):
        ok = [ms for p, ms, _ in res[name] if p is not None]
        vers = sorted({v for _, _, v in res[name] if v})
        out[name] = {"url": prod_url if name == "prod" else cand_url, "errors": len(rows) - len(ok),
                     "model_versions": vers, **_pct(ok)}
    return out

def format_shadow(out):
    """Return the shadow result as readable text."""
    lines = ["== Shadow comparison: same requests to prod and candidate ==",
             f"requests: {out['n_requests']}  compared: {out['n_compared']}",
             f"(i)   mean |prod - candidate|      = {out['mean_abs_diff_min']} min",
             f"(ii)  requests with |diff| > 10 min = {out['pct_diff_gt_10min']} %",
             f"      max |diff|                    = {out['max_abs_diff_min']} min",
             f"(iii) {'server':<10}{'version':>9}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'errors':>8}"]
    for name in ("prod", "candidate"):
        s = out[name]
        lines.append(f"      {name:<10}{','.join(s['model_versions']) or '-':>9}{s['p50_ms']:>9}"
                     f"{s['p95_ms']:>9}{s['p99_ms']:>9}{s['errors']:>8}")
    return "\n".join(lines)

# Command line
def _write(text, out):
    """Print the text. Also write it to the file "out" if it is given."""
    print(text)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            f.write(text + "\n")

def _header(title):
    """Return a header with the time and the machine description."""
    return (f"# {title}\n# generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"# machine: {json.dumps(machine_info())}\n# tracking: {tracking_uri()}\n")

def main(argv=None):
    ap = argparse.ArgumentParser(description="SwiftParcel ETA -- MLflow registry, gate and shadow.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register", help="log and register v1 and v2; set the aliases")
    g = sub.add_parser("gate", help="run the promotion gate (default and min_rel_gain=0.60)")
    g.add_argument("--out")
    s = sub.add_parser("shadow", help="compare two running servers on the same requests")
    s.add_argument("--prod-url", default="http://localhost:5101/invocations")
    s.add_argument("--cand-url", default="http://localhost:5102/invocations")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--out")
    a = ap.parse_args(argv)

    if a.cmd == "register":
        register()
    elif a.cmd == "gate":
        setup_mlflow()
        parts = [_header("Part B2 -- promotion gate"),
                 "Registry:\n" + format_registry_table(registry_table()) + "\n"]
        results = {}
        for title, th in [("default thresholds", {}), ("min_rel_gain=0.60", {"min_rel_gain": 0.60})]:
            results[title] = promotion_gate(**th)
            parts.append(format_gate(results[title], title) + "\n")
        parts.append("JSON:\n" + json.dumps(results, indent=2))
        _write("\n".join(parts), a.out)
    elif a.cmd == "shadow":
        out = shadow(a.prod_url, a.cand_url, n=a.n)
        _write("\n".join([_header("Part B3 -- shadow comparison"), format_shadow(out),
                          "\nJSON:\n" + json.dumps(out, indent=2)]), a.out)

if __name__ == "__main__":
    main()
