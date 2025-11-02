# Requirements Specification

This document captures the functional and non-functional requirements for FIWARE Chronos.

## Functional Requirements

- Model management
  - Create, retrieve, update, and delete forecasting model definitions.
  - List supported model types (LSTM, GRU).
  - Validate hyperparameters and FIWARE metadata.
- Training lifecycle
  - Launch training jobs per model with optional history window selection.
  - Collect historical data from STH‑Comet with pagination (hLimit/hOffset).
  - Preprocess time series and persist artifacts (model, scalers, metadata) to GridFS.
  - Track job status, timings, and metrics (loss, MAE, RMSE, R2).
  - Cancel training jobs and clean up partially created resources.
- Prediction
  - Execute on‑demand forecasts for a trained model/training job.
  - Publish forecasts as NGSI attributes to Orion.
  - Retrieve prediction history from STH‑Comet.
- Recurring forecasts
  - Enable/disable periodic forecasts per training job.
  - Celery beat schedules evaluation and enqueues executions at the right cadence.
- FIWARE integration
  - Discover devices and attributes from IoT Agent.
  - Ensure service groups, devices, and Orion entities/subscriptions when enabling forecasts.
- System
  - Health endpoints expose MongoDB, RabbitMQ, Redis, and FIWARE connectivity.
  - Info endpoint returns build, configuration summary, and start time.

## Non‑Functional Requirements

- Reliability & Resilience
  - Task retries for transient failures; at‑least‑once job execution semantics.
  - Idempotent publishing to Orion where possible.
- Performance
  - Batch data collection to minimize STH‑Comet round‑trips.
  - Configurable worker concurrency and dataset chunk sizes.
- Scalability
  - Horizontal scale of API and Celery workers.
  - Broker‑based decoupling for long‑running tasks.
- Security
  - Support API protection via reverse proxies (TLS/OAuth2/JWT) in front of FastAPI.
  - Secrets managed via environment variables and Docker/Kubernetes secrets.
- Observability
  - Structured logging compatible with Loki/ELK.
  - Health checks and metrics endpoints for basic status.
- Portability
  - Docker Compose reference; Kubernetes‑friendly configuration.
- Quality
  - Automated tests (unit, integration, functional, e2e) with coverage ≥ 90%.

## Constraints & Assumptions

- NGSI‑v2 for Orion and STH‑Comet integration; NGSI‑LD is planned.
- MongoDB is the system of record; artifacts stored in GridFS.
- RabbitMQ is the default Celery broker, Redis as result backend.
