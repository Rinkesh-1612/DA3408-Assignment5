"""
verify_ray.py: AIOps Module 4, Assignment 5, Part C2: verify the sticky canary on a 2-CPU Ray.

The script deploys eta_ray.build() and checks five properties with assertions:
    (a) With weight 0.2, the share of 1000 customers on the candidate is 0.2 +/- 4 percentage points.
    (b) 200 customers get the same arm on a second call.
    (c) The arms stay the same after a redeploy of the identical application.
    (d) After the weight goes from 0.2 to 0.5, no customer moves from v2 back to v1.
    (e) The header "X-Beta: 1" sends 50 requests from v1 customers to v2.
The script also prints evidence: replica counts, the autoscaling config, the CPU reservations,
the replicas that answered, and the machine.

Run from the repository root:
    python partC/verify_ray.py --out partC/verify_output.txt
Exit code: 0 if all checks pass, 1 if one check fails.

Citation: the start / wait / delete-before-redeploy steps follow the lecture notebook
"Lecture 10 -- Ray_Serve_Exercise.ipynb" (Lab2/DA3408) and its cheat sheet.
"""
import argparse
import collections
import concurrent.futures as cf
import os
import platform
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)) # Let "import eta_ray" work from any directory.

APP_NAME = "eta-canary"
PREDICT_PATH = "/v1/models/eta:predict"
ROW = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2,
       "traffic_index": 0.4, "rain": 0, "weight_kg": 6.0}

# Output helpers
class Tee:
    """Write each line to the terminal and to a file."""

    def __init__(self, path):
        self.f = open(path, "w") if path else None

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()

class Checks:
    """Record the result of each check. Print PASS or FAIL with evidence."""

    def __init__(self, log):
        self.log, self.results = log, []

    def check(self, label, ok, evidence):
        self.results.append((label, bool(ok)))
        self.log(f"[{'PASS' if ok else 'FAIL'}] {label}: {evidence}")
        return ok

def machine_info():
    """Return a short description of this machine."""
    mem = "unknown"
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
            mem = f"{kb / 1024 ** 2:.1f} GiB"
    except Exception:
        pass
    return (f"python {platform.python_version()} | {platform.system()} {platform.release()} | "
            f"host CPUs visible: {os.cpu_count()} | RAM: {mem}")

# HTTP client helpers
_session_by_thread = {}

def _session():
    import threading
    key = threading.get_ident()
    if key not in _session_by_thread:
        _session_by_thread[key] = requests.Session()
    return _session_by_thread[key]

def call(url, customer_id, beta=False):
    """Send one request for one customer. Return the JSON body."""
    headers = {"X-Beta": "1"} if beta else {}
    r = _session().post(url, json={"customer_id": customer_id, "instances": [ROW]}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def route_all(url, ids, beta=False, workers=8):
    """Send one request for each customer. Return {customer_id: response body}."""
    with cf.ThreadPoolExecutor(workers) as ex:
        bodies = list(ex.map(lambda c: call(url, c, beta), ids))
    return dict(zip(ids, bodies))

def wait_ready(base, timeout=120):
    """Poll /healthz until it answers 200. Return the seconds it took."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{base}/healthz", timeout=2).status_code == 200:
                return round(time.time() - t0, 1)
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{base}/healthz is not ready after {timeout}s")

# Ray / Serve helpers
def deploy(serve, eta_ray, weight, base, log):
    """Deploy the application with one weight. Wait until the router answers."""
    serve.run(eta_ray.build(weight=weight), name=APP_NAME, route_prefix="/")
    ready = wait_ready(base)
    log(f"deployed weight={weight} (ready after {ready}s)")

def delete_app(serve, log, wait_s=5):
    """Delete the application. Wait before a new deploy (lecture gotcha with autoscaled apps)."""
    serve.delete(APP_NAME)
    time.sleep(wait_s)
    log(f"deleted application '{APP_NAME}' and waited {wait_s}s")

def print_status(serve, eta_ray, log):
    """Print the replica states of each deployment."""
    status = serve.status().applications[APP_NAME]
    for name, d in status.deployments.items():
        log(f"  deployment {name:12s} status={d.status} replicas={dict(d.replica_states)}")

def print_resources(ray, serve, eta_ray, log):
    """Print the CPU reservation of each deployment and of the Ray cluster."""
    total = ray.cluster_resources().get("CPU", 0)
    free = ray.available_resources().get("CPU", 0)
    log(f"  ray cluster CPU={total}  available CPU={free:.2f}  reserved now={total - free:.2f}")
    cand_max = eta_ray.CAND_AUTOSCALING["max_replicas"]
    log(f"  reservation: router {eta_ray.ROUTER_CPUS} + prod {eta_ray.PROD_REPLICAS} x {eta_ray.MODEL_CPUS}"
        f" + candidate (max {cand_max}) x {eta_ray.MODEL_CPUS} = {eta_ray.MAX_TOTAL_CPUS:.2f} CPU (limit 2.0)")

# Main verification
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", help="also write all evidence to this file")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP host for Serve (0.0.0.0 in a container)")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--n-customers", type=int, default=1000)
    ap.add_argument("--weight", type=float, default=0.2)
    ap.add_argument("--new-weight", type=float, default=0.5)
    a = ap.parse_args()

    log = Tee(a.out)
    checks = Checks(log)
    log("=" * 78)
    log("Part C2 -- sticky canary verification (Ray Serve)")
    log("machine:", machine_info())

    # Make the partC folder importable in the Ray worker processes too.
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [str(HERE), os.environ.get("PYTHONPATH", "")] if p)
    import ray
    from ray import serve
    import eta_ray

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    serve.start(http_options={"host": a.host, "port": a.port})
    base = f"http://127.0.0.1:{a.port}"
    url = base + PREDICT_PATH
    log(f"ray {ray.__version__}: cluster with num_cpus=2; serve HTTP on {a.host}:{a.port}")
    log(f"candidate autoscaling_config: {eta_ray.CAND_AUTOSCALING}")
    log(f"bucket function: sha256('{eta_ray.SALT}:' + customer_id)[:8 bytes] / 2**64 ; v2 if bucket < weight")

    try:
        ids = [f"cust-{i:05d}" for i in range(a.n_customers)]

        # Deploy with weight 0.2
        log("-" * 78)
        deploy(serve, eta_ray, a.weight, base, log)
        print_status(serve, eta_ray, log)
        print_resources(ray, serve, eta_ray, log)

        # (a) Share of customers on the candidate
        log("-" * 78)
        first = route_all(url, ids)
        arm1 = {c: b["arm"] for c, b in first.items()}
        n_v2 = sum(v == "v2" for v in arm1.values())
        share = n_v2 / len(ids)
        se = (a.weight * (1 - a.weight) / len(ids)) ** 0.5
        expected = sum(eta_ray.choose_arm(c, a.weight) == "v2" for c in ids)
        checks.check("(a) candidate share within +/-4 pp of weight",
                     abs(share - a.weight) <= 0.04,
                     f"{n_v2}/{len(ids)} customers on v2 = {share:.3f} (target {a.weight}, "
                     f"binomial SE {se * 100:.1f} pp, offline choose_arm() predicts {expected})")
        mism = [c for c, b in first.items() if b["model_version"] != b["arm"]]
        replicas = collections.defaultdict(set)
        for b in first.values():
            replicas[b["arm"]].add(b["served_by"])
        checks.check("(a') model_version matches arm for every response", not mism,
                     f"{len(mism)} mismatches; replicas that answered: "
                     + ", ".join(f"{k}={len(v)}" for k, v in sorted(replicas.items())))

        # (b) Same arm on a second call
        log("-" * 78)
        sample = ids[:200]
        second = route_all(url, sample)
        changed = [c for c in sample if second[c]["arm"] != arm1[c]]
        checks.check("(b) 200 customers keep the same arm on a second call", not changed,
                     f"{len(sample) - len(changed)}/200 unchanged "
                     f"({sum(arm1[c] == 'v2' for c in sample)} of them on v2)")

        # (c) Same arm after a redeploy of the identical app
        log("-" * 78)
        delete_app(serve, log)
        deploy(serve, eta_ray, a.weight, base, log)
        third = route_all(url, ids)
        changed_200 = [c for c in sample if third[c]["arm"] != arm1[c]]
        changed_all = [c for c in ids if third[c]["arm"] != arm1[c]]
        checks.check("(c) same arms after redeploying the identical app", not changed_200 and not changed_all,
                     f"200-sample: {200 - len(changed_200)}/200 unchanged; "
                     f"all customers: {len(ids) - len(changed_all)}/{len(ids)} unchanged")

        # (d) Raise the weight: no customer moves back from v2 to v1
        log("-" * 78)
        deploy(serve, eta_ray, a.new_weight, base, log) # Only the router argument changes.
        fourth = route_all(url, ids)
        arm4 = {c: b["arm"] for c, b in fourth.items()}
        back = [c for c in ids if arm1[c] == "v2" and arm4[c] == "v1"]
        moved_up = sum(arm1[c] == "v1" and arm4[c] == "v2" for c in ids)
        share4 = sum(v == "v2" for v in arm4.values()) / len(ids)
        checks.check(f"(d) weight {a.weight} -> {a.new_weight}: no customer moves v2 -> v1", not back,
                     f"v2->v1 moves: {len(back)}; v1->v2 moves: {moved_up}; "
                     f"stayed on v2: {n_v2 - len(back)}/{n_v2}; new v2 share = {share4:.3f}")
        checks.check(f"(d') new share within +/-4 pp of {a.new_weight}", abs(share4 - a.new_weight) <= 0.04,
                     f"{share4:.3f}")
        print_status(serve, eta_ray, log)

        # (e) Header override
        log("-" * 78)
        v1_customers = [c for c in ids if arm4[c] == "v1"][:50]
        forced = route_all(url, v1_customers, beta=True)
        not_v2 = [c for c, b in forced.items() if b["arm"] != "v2" or b["model_version"] != "v2"]
        again = route_all(url, v1_customers[:10])
        checks.check("(e) X-Beta: 1 forces the candidate for 50 v1 customers",
                     len(v1_customers) == 50 and not not_v2,
                     f"{50 - len(not_v2)}/50 answered by v2; without the header the same customers "
                     f"go back to {sorted({b['arm'] for b in again.values()})}")
    except Exception as exc: # Report the failure, then clean up.
        checks.check("unexpected error", False, f"{type(exc).__name__}: {exc}")
    finally:
        log("-" * 78)
        try:
            serve.delete(APP_NAME)
            serve.shutdown()
        finally:
            ray.shutdown()

    passed = sum(ok for _, ok in checks.results)
    log(f"SUMMARY: {passed}/{len(checks.results)} checks passed")
    log.close()
    return 0 if passed == len(checks.results) else 1

if __name__ == "__main__":
    sys.exit(main())
