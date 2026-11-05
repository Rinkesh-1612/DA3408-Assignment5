# Makefile -- SwiftParcel ETA service (AIOps Module 4, Assignment 5).
#
# Docker path (recommended):  make build  then  make all
# Local path (no Docker):     make venv data test-local
#
# The Compose project name "da-aiops-a5" keeps these containers apart from other stacks.
# Run the parts one after the other. Parallel runs share the CPUs and change the latency numbers.

PROJECT  ?= da-aiops-a5
COMPOSE  := docker compose -p $(PROJECT)
PY       ?= .venv/bin/python

.PHONY: help venv data test-local build up down test loadtest probe mutation partA partB partC partD all logs clean zip

help:
	@echo "make build     Build the two images (api, tools). The build also makes the data and models."
	@echo "make up        Start the Part A service (v1) on http://localhost:8010"
	@echo "make test      Run all pytest suites in a container (Part A and Part B)"
	@echo "make loadtest  Part A3: load test 1 worker and 2 workers (3 variants) -> partA/loadtest_results.json"
	@echo "make mutation  Part A2: change the code in 5 ways, each change must fail a test -> partA/mutation_output.txt"
	@echo "make probe     Part A3: compare new and kept connections -> partA/keepalive_probe.txt"
	@echo "make partB     Part B: register, gate, serve prod:5101 cand:5102, shadow -> partB/*.txt"
	@echo "make partC     Part C: Ray Serve sticky canary verification -> partC/verify_output.txt"
	@echo "make partD     Part D: smoke test (v1 pass, v2 pass, wrong version must fail) -> partD/smoke_output.txt"
	@echo "make all       Run build, test, loadtest, partB, partC and partD in this order"
	@echo "make down      Stop and remove all containers of this project"
	@echo "make clean     Do 'make down', remove the MLflow volume and the local generated files"
	@echo "make zip ROLL=<rollno>  Make Module4_<rollno>.zip for submission (no data, no models)"

# local path (no Docker)
venv:
	python3 -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

data:
	$(PY) eta_data.py
	$(PY) -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

test-local:
	$(PY) -m pytest -q partA/test_eta_app.py partB/test_gate.py

# Docker path
build:
	$(COMPOSE) --profile release --profile scale --profile mlflow --profile jobs build

up:
	$(COMPOSE) up -d --wait api
	@curl -s http://localhost:8010/healthz && echo

down:
	$(COMPOSE) --profile release --profile scale --profile mlflow --profile jobs down --remove-orphans

test:
	$(COMPOSE) run --rm tests

loadtest partA:
	$(COMPOSE) --profile scale up -d --wait api api-2w api-2w-asyncio api-2w-asyncio-fixed
	$(COMPOSE) run --rm loadtest
	$(COMPOSE) --profile scale stop api-2w api-2w-asyncio api-2w-asyncio-fixed

partB:
	$(COMPOSE) --profile mlflow --profile jobs down -v --remove-orphans
	$(COMPOSE) --profile mlflow up -d --wait mlflow-prod mlflow-cand
	$(COMPOSE) logs mlflow-register | tail -n 40
	$(COMPOSE) run --rm shadow
	$(COMPOSE) --profile mlflow stop mlflow-prod mlflow-cand

mutation:
	$(COMPOSE) run --rm mutation

probe:
	$(COMPOSE) --profile scale up -d --wait api api-2w api-2w-asyncio api-2w-asyncio-fixed
	$(COMPOSE) run --rm probe
	$(COMPOSE) --profile scale stop api-2w api-2w-asyncio api-2w-asyncio-fixed

partC:
	$(COMPOSE) run --rm ray-verify

partD:
	$(COMPOSE) --profile release up -d --wait api api-v2
	@{ echo "### 1. v1 service, expect v1 (must pass)"; \
	   $(COMPOSE) run --rm -T smoke; echo "exit code: $$?"; echo; \
	   echo "### 2. v2 service, expect v2 (must pass)"; \
	   $(COMPOSE) run --rm -T smoke python partD/smoke_eta.py --url http://api-v2:8000 --expect-version v2 --max-p95-ms 80; echo "exit code: $$?"; echo; \
	   echo "### 3. v1 service, expect v2 (must fail: the wrong version is deployed)"; \
	   $(COMPOSE) run --rm -T smoke python partD/smoke_eta.py --url http://api:8000 --expect-version v2 --max-p95-ms 80; echo "exit code: $$?"; \
	} 2>&1 | tee partD/smoke_output.txt
	$(COMPOSE) --profile release stop api-v2

all: build test mutation loadtest probe partB partC partD
	@echo "All parts ran. See partA/, partB/, partC/ and partD/ for the outputs."

logs:
	$(COMPOSE) --profile release --profile scale --profile mlflow --profile jobs logs --tail 50

clean: down
	-docker volume rm $(PROJECT)_mlflow-state
	rm -rf data mlflow.db mlruns mlartifacts .pytest_cache partA/__pycache__ partB/__pycache__ partC/__pycache__ partD/__pycache__

# submission (assignment section 7)
# Make Module4_<rollno>.zip. It holds the report, the four part folders and the files to run them.
# It holds no data, no models and no databases (eta_data.py makes them again).
zip:
	@test -n "$(ROLL)" || { echo "Use: make zip ROLL=<rollno>"; exit 1; }
	find partA partB partC partD -name __pycache__ -prune -exec rm -rf {} +
	rm -f Module4_$(ROLL).zip
	python3 -m zipfile -c Module4_$(ROLL).zip REPORT.pdf README.md AI_DISCLOSURE.md partA partB partC partD \
		eta_data.py load_test.py requirements.txt Dockerfile compose.yaml Makefile .dockerignore
	python3 -m zipfile -l Module4_$(ROLL).zip
