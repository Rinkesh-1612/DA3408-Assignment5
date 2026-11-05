"""
run_loadtest.py: Part A3 experiments. This script uses the given load generator (load_test.run).

Experiments:
    1. Concurrency sweep 1, 2, 4, 8, 16 (4 s each, batch 1) against one uvicorn worker (v1).
    2. Cost per row: batch 1 against batch 100 at concurrency 1.
    3. The capacity change: the same sweep against each --variant service (two workers).
The sweeps of all services are interleaved (see sweep_interleaved). Each sweep runs --repeats times.
The analysis uses the median of the repeats at each concurrency level.

Run (Docker):  make loadtest
Run (venv):    python partA/run_loadtest.py --url http://localhost:8010/v1/models/eta:predict \
                   --variant workers2=http://localhost:8011/v1/models/eta:predict --out partA/loadtest_results.json

The script does a warm-up before each experiment. Latency is reported as p50, p95 and p99.
Little's Law check (closed loop): concurrency N = throughput X * mean latency W.
The load generator gives p50, not the mean. Thus we show X * p50 as an estimate of N.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import load_test # noqa: E402  (the given generator)

SLO_P95_MS = 80.0

# helpers
def base_url(predict_url: str) -> str:
    """Remove the predict route from a URL. Keep the scheme, host and port."""
    return predict_url.split("/v1/models/")[0]

def wait_ready(predict_url: str, timeout: float = 60.0) -> None:
    """Poll /healthz until it gives 200. Stop with an error after the timeout."""
    url, t0 = base_url(predict_url) + "/healthz", time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} is not ready after {timeout:.0f} s")

def cgroup_cpu_limit() -> str:
    """Read the CPU limit of this container (cgroup v2). Give "none" if there is no limit."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return "none" if quota == "max" else f"{int(quota) / int(period):g} CPUs"
    except (OSError, ValueError):
        return "unknown"

def machine_info() -> dict:
    """Collect facts about the machine. Hardware numbers depend on these facts."""
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus_visible": os.cpu_count(), "generator_cpu_limit": cgroup_cpu_limit()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        mem_kb = int(Path("/proc/meminfo").read_text().split()[1])
        info["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return info

def load_avg() -> list:
    """Give the 1, 5 and 15 minute load averages of the host. They show other load on the CPUs."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []

def add_littles_law(row: dict) -> dict:
    """Add the Little's Law estimate N = X * W, with W = p50 (in seconds)."""
    row["littles_N_est"] = round(row["rps"] * row["p50_ms"] / 1000.0, 2)
    return row

# experiments
def sweep_interleaved(services: dict, rows: list, levels: list, duration: float, repeats: int) -> dict:
    """Run the concurrency sweep (batch 1) for all services, "repeats" times.
    Interleave the services: at each level, run each service one after the other.
    Thus a slow period on the shared host hits all services, not only one of them.
    Give one list of result rows for each service label."""
    for url in services.values(): # Warm-up. Discard the results.
        load_test.run(url, "rest", 2, 2, rows, 1)
    out = {label: [] for label in services}
    for rep in range(1, repeats + 1):
        for c in levels:
            for label, url in services.items():
                r = add_littles_law(load_test.run(url, "rest", c, duration, rows, 1))
                r["repeat"] = rep
                r["host_load_1m"] = load_avg()[:1]
                out[label].append(r)
    for label, res in out.items():
        print(f"\n== Sweep: {label} ==")
        print(pd.DataFrame(res).drop(columns=["versions"]).to_string(index=False))
    return out

def batch_cost(url: str, rows: list, duration: float, repeats: int = 1) -> list:
    """Measure the cost per row for batch 1 and batch 100 at concurrency 1.
    Run each batch size "repeats" times, alternating the sizes. Report each run and the median."""
    load_test.run(url, "rest", 1, 2, rows, 100) # Warm-up for the large batch.
    out = []
    for rep in range(1, repeats + 1):
        for b in (1, 100):
            r = load_test.run(url, "rest", 1, duration, rows, b)
            r.update(batch=b, repeat=rep, rows_per_s=round(r["rps"] * b, 1),
                     ms_per_row_p50=round(r["p50_ms"] / b, 4),
                     ms_per_row_throughput=round(1000.0 / (r["rps"] * b), 4) if r["rps"] else None)
            out.append(r)
    df = pd.DataFrame(out)
    print("\n== Cost per row (concurrency 1) ==")
    cols = ["repeat", "batch", "requests", "rps", "rows_per_s", "p50_ms", "p95_ms", "ms_per_row_p50",
            "ms_per_row_throughput", "errors"]
    print(df[cols].to_string(index=False))
    print("median:")
    print(df.groupby("batch")[["rps", "rows_per_s", "p50_ms", "p95_ms", "ms_per_row_p50",
                               "ms_per_row_throughput"]].median().to_string())
    return out

def analyse(rows: list) -> dict:
    """Find the saturation point and the capacity under the SLO (p95 <= 80 ms).
    Use the median over the repeats for each concurrency level.

    Saturation, two ways:
      knee_littles_law  N* = X_max * W(1). W(1) is the latency with one client (no queue).
                        Above N*, Little's Law says that extra clients can only add queueing delay.
      plateau_start     The first level where doubling the clients adds less than 25 % throughput.
    """
    df = pd.DataFrame(rows).groupby("concurrency")[["rps", "p50_ms", "p95_ms", "p99_ms", "errors"]].median()
    df = df.reset_index().sort_values("concurrency")
    peak = float(df.rps.max())
    w1_s = float(df.p50_ms.iloc[0]) / 1000.0
    gains = df.rps.pct_change().shift(-1) # Throughput gain at the next level.
    flat = df[gains < 0.25]
    plateau = int(flat.concurrency.iloc[0]) if len(flat) else int(df.concurrency.iloc[-1])
    ok = df[(df.p95_ms <= SLO_P95_MS) & (df.errors == 0)]
    best = ok.loc[ok.rps.idxmax()] if len(ok) else None
    return {"median_table": df.round(2).to_dict(orient="records"), "peak_rps": peak,
            "knee_littles_law_N": round(peak * w1_s, 2), "plateau_start_concurrency": plateau,
            "slo_p95_ms": SLO_P95_MS,
            "capacity_under_slo_rps": float(best.rps) if best is not None else None,
            "capacity_under_slo_at_concurrency": int(best.concurrency) if best is not None else None}

# main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="predict URL of the 1-worker service (v1)")
    ap.add_argument("--variant", action="append", default=[], metavar="LABEL=URL",
                    help="extra service to sweep, e.g. workers2=http://api-2w:8000/v1/models/eta:predict")
    ap.add_argument("--repeats", type=int, default=3, help="run each sweep this many times")
    ap.add_argument("--data", default=str(ROOT / "data" / "eta_sample.csv"))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--out", default=str(ROOT / "partA" / "loadtest_results.json"))
    a = ap.parse_args()

    rows = pd.read_csv(a.data).to_dict(orient="records")
    wait_ready(a.url)
    result = {"machine": machine_info(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host_load_avg_start": load_avg(), "duration_s": a.duration, "url": a.url}

    services = {"1worker": a.url}
    for spec in a.variant:
        label, url = spec.split("=", 1)
        services[label] = url
    for url in services.values():
        wait_ready(url)

    sweeps = sweep_interleaved(services, rows, a.concurrency, a.duration, a.repeats)
    for label, res in sweeps.items():
        result[f"sweep_{label}"] = res
        result[f"analysis_{label}"] = analyse(res)
    one = result["analysis_1worker"]
    for label in services:
        if label == "1worker":
            continue
        two = result[f"analysis_{label}"]
        result[f"capacity_change_{label}"] = {
            "peak_rps_ratio": round(two["peak_rps"] / one["peak_rps"], 2),
            "slo_capacity_ratio": (round(two["capacity_under_slo_rps"] / one["capacity_under_slo_rps"], 2)
                                   if one["capacity_under_slo_rps"] and two["capacity_under_slo_rps"] else None)}
    result["batch_cost"] = batch_cost(a.url, rows, a.duration, a.repeats)

    result["host_load_avg_end"] = load_avg()
    print("\n== Analysis ==")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("capacity") or k.startswith("analysis")}, indent=2))
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {a.out}")

if __name__ == "__main__":
    main()
