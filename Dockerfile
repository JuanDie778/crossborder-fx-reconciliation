# Multi-stage Dockerfile for crossborder-fx-reconciliation
# Author: Juan Diego Ussa Aponte (Valencia, Spain)

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal Runtime Environment
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged application user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy domain modules, storage, engines, tests, and documentation
COPY --chown=appuser:appgroup domain_ledger.py sqlite_ledger_storage.py fx_risk_and_routing.py swift_cashmatch_engine.py fx_arbitrage_reconciliation_engine.py ./
COPY --chown=appuser:appgroup test_flywire_poc.py ./
COPY --chown=appuser:appgroup dashboard_reconciliation.html README.md requirements.txt ./

USER appuser

# Default execution: run the institutional cross-border FX & reconciliation benchmark
# Can be overridden at runtime with `pytest test_flywire_poc.py -v`
CMD ["python", "fx_arbitrage_reconciliation_engine.py"]
