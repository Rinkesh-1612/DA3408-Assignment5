# Dockerfile: SwiftParcel ETA service (AIOps Module 4, Assignment 5).
#
# The file has two targets. Both targets use the same packages and the same data.
#   api   : Part A (REST service) and Part D (smoke test).
#   tools : all parts. It adds Part B (MLflow) and Part C (Ray Serve).
#
# Build:
#   docker build --target api   -t swiftparcel-eta:api   .
#   docker build --target tools -t swiftparcel-eta:tools .
#
# The build runs eta_data.py. The data seeds are fixed.
# Thus each build makes the same data and the same model files.
# The repository does not keep model files or databases.
# Idea from the course: Lecture 5 (multi-stage builds) and Module 4 Dockerfile.serve.

ARG PYTHON_IMAGE=python:3.12-slim

# Stage 1: install the packages
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Copy the requirement file before the code. Docker then keeps this layer in the cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: make the base image with the packages and the data
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true
RUN useradd --create-home --uid 1000 eta
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY eta_data.py load_test.py ./
# Make the data and the two model files. Make the sample of 200 rows for the load test.
RUN python eta_data.py && \
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

# Target: api
FROM base AS api
# Copy the code last. A change in the code then does not run the steps above again.
COPY partA/ partA/
COPY partD/ partD/
RUN chown -R eta:eta /app
USER eta
ENV ETA_MODEL=data/eta_v1.joblib PYTHONPATH=/app:/app/partA
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "eta_app:app", "--app-dir", "partA", "--host", "0.0.0.0", "--port", "8000"]

# Target: tools
FROM base AS tools
COPY partA/ partA/
COPY partB/ partB/
COPY partC/ partC/
COPY partD/ partD/
RUN mkdir -p /state && chown -R eta:eta /app /state
USER eta
ENV PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=sqlite:////state/mlflow.db \
    ETA_ARTIFACT_ROOT=/state/mlartifacts
CMD ["bash"]
