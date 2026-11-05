# DA3408 Assignment 5: serving a delivery-time model four ways

Rinkesh (DA24B017) · AI Operations, Module 4

SwiftParcel, a made-up courier, wants to predict how many minutes a delivery takes. Two models are already trained: **v1** (Ridge regression) and **v2** (gradient boosting). I turned them into a service in four parts:

- **A, REST service (FastAPI):** checks its input, tags every answer with a request ID, exposes Prometheus metrics, and has tests and a load test.
- **B, MLflow:** both models sit in a registry as `prod` and `candidate`. A promotion gate and a shadow test decide if v2 can replace v1.
- **C, Ray Serve:** a canary where the same customer always gets the same model version.
- **D, smoke test:** a script that exits with 0 to allow a release and 1 to block it.

The explanations, the tables and the list of what I did not test are in `REPORT.pdf`. This README only shows how to run things.

**Demo video (about 5 minutes):** https://drive.google.com/file/d/1ubb1-eYKJDFofM-4JvTyACioOVNjIW1R/view?usp=drive_link

## What you need

Docker with the Compose plugin. You don't need Python or a GPU. Plan for about 3 GB of disk and 2 free CPU cores (every container is capped at 2 CPUs on purpose, because the assignment targets a 2-core laptop; my own machine has 8 cores and 23 GB of RAM). Ports 8010 to 8014, 5101 and 5102 must be free. If one is taken, change the left number of that port in `compose.yaml`.

## Run it

```bash
git clone https://github.com/Rinkesh-1612/DA3408-Assignment5.git
cd DA3408-Assignment5
make build     # slow the first time (a few minutes): Docker installs Ray and MLflow and trains the two models
make test      # should end with "53 passed"
make up        # starts the Part A service on localhost:8010
```

Then try it:

```bash
curl -s -X POST localhost:8010/v1/models/eta:predict -H 'content-type: application/json' -H 'X-Request-ID: demo-001' \
     -d '{"instances":[{"distance_km":12.5,"n_stops":4,"hour":10,"weekday":2,"traffic_index":0.3,"rain":0,"weight_kg":8.0}]}'
```

You should get `{"predictions":[42.5],"model_version":"v1","request_id":"demo-001"}`. Bad input (for example `"hour": 25`) gives a 422 with the list of wrong fields, and the request ID comes back in both cases.

Now run the parts. **Run them one at a time.** They share the CPUs, so running two together ruins the latency numbers (one of my early load tests failed that way and I threw it away).

| Command | What it does | Output | Time |
|---|---|---|---|
| `make loadtest` | Part A load test (1 worker against 2 workers) | `partA/loadtest_results.json` | about 8 min |
| `make mutation` | breaks the service code in 5 ways, each must fail a test | `partA/mutation_output.txt` | under 1 min |
| `make probe` | compares new and kept connections | `partA/keepalive_probe.txt` | 1 to 2 min |
| `make partB` | registers the models, runs the gate, runs the shadow test | `partB/gate_output.txt`, `partB/shadow_output.txt` | 1 to 2 min |
| `make partC` | Ray canary checks | `partC/verify_output.txt` | 1 to 2 min |
| `make partD` | smoke test three times, and the third one must fail (exit 1) | `partD/smoke_output.txt` | about 1 min |

Only the load test time is measured (7.6 minutes). The others are my rough feeling. `make all` runs everything and takes roughly 15 minutes after the build. `make down` stops the containers, `make clean` also deletes the generated data and the MLflow volume, and `make zip ROLL=<rollno>` builds the submission zip.

## Results

- 53 tests pass, and all 5 deliberate breaks of the code are caught.
- Capacity under the SLO (p95 ≤ 80 ms): 83 requests per second with 1 worker, and 161 with 2 workers (1.93×).
- v2 halves the error (MAE 11.96 to 6.11 minutes). The gate promotes it with the default thresholds and blocks it when I ask for a 60 % gain.
- In the canary, 20.2 % of 1000 customers get v2 (I asked for 20 %), and nobody changes version on a second call or a redeploy.
- The smoke test passes 8 of 8 checks on v1 and v2, and blocks a deploy with the wrong version.

## Where things are

```
partA/   eta_app.py (the service), test_eta_app.py, run_loadtest.py, serve_workers.py,
         mutation_check.py, probe_keepalive.py, and the output files of the load test, mutation check and probe
partB/   eta_mlflow.py (registry, gate, shadow), test_gate.py, gate_output.txt, shadow_output.txt
partC/   eta_ray.py (the canary), verify_ray.py, verify_output.txt
partD/   smoke_eta.py, smoke_output.txt
eta_data.py, load_test.py     given by the course, unchanged
Dockerfile, compose.yaml, Makefile, requirements.txt (the only requirements file)
REPORT.pdf                    the report (tables, written answers, sources, list of what I did not test)
AI_DISCLOSURE.md              how I used AI tools
```

## Without Docker

`make venv`, then `source .venv/bin/activate`, then `make data` and `make test-local` work with Python 3.10 or later. The servers for Parts B and C you will have to start by hand, and the commands are in `compose.yaml`. I only tested the Docker path end to end, so I can't promise the rest.

## If something breaks

- **`port is already allocated`:** change the left port number in `compose.yaml`, or stop the program that uses the port.
- **Latency numbers jump around:** something else uses your CPUs. Run one part at a time, and compare medians.
- **Ray is slow or shows strange errors:** run `make down`, then `make partC` again.
- **The old model version still answers after you move an alias:** a running MLflow server reads its alias only at start-up, so restart it.
- **A 422 on a request that looks right:** check that the top key is `instances`, that every value is in range, and that you don't have a delivery with more than 150 km and more than 20 stops.

I tested this on Linux 6.8 with Docker 29.9 only.

## Credits and AI usage

`eta_data.py` and `load_test.py` come from the course. Ideas and small parts of the code come from the Module 4 lecture notebooks, and `REPORT.pdf` lists exactly which. I built this together with an AI coding assistant (Claude Code), and `AI_DISCLOSURE.md` explains how.
