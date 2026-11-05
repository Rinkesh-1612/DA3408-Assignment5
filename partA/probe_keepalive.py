"""probe_keepalive.py: measure the latency of one request, with and without a kept connection.

Run:
    python partA/probe_keepalive.py --url http://localhost:8010 --url http://localhost:8013

Why this script exists:
The plain "uvicorn --workers 2" service was slow for each request, also with one client.
This script shows the cause. A request on a new connection takes about the same time on all services.
A request on a kept connection (keep-alive) takes about 40 ms longer on the slow service.
The cause is a missing TCP_NODELAY on the worker sockets. See serve_workers.py.
The problem occurs only with the asyncio event loop. The uvloop event loop does not have it.
"""
import argparse
import time

import requests

BODY = {"instances": [{"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
                       "traffic_index": 0.3, "rain": 0, "weight_kg": 8.0}]}
ROUTE = "/v1/models/eta:predict"


def time_requests(url, keep_connection, count):
    """Send the request "count" times. Give the sorted times in milliseconds."""
    session = requests.Session() if keep_connection else requests
    times = []
    for _ in range(count + 10):
        start = time.perf_counter()
        session.post(url + ROUTE, json=BODY, timeout=10)
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times[10:])


def main():
    parser = argparse.ArgumentParser(description="Compare new and kept connections.")
    parser.add_argument("--url", action="append", required=True, help="base URL, one or more")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    print(f"{'service':36s} {'connection':12s} {'p50 ms':>8s} {'p95 ms':>8s}")
    for url in args.url:
        for keep, name in ((False, "new"), (True, "kept")):
            times = time_requests(url, keep, args.count)
            print(f"{url:36s} {name:12s} {times[len(times) // 2]:8.1f} {times[int(0.95 * len(times))]:8.1f}")


if __name__ == "__main__":
    main()
