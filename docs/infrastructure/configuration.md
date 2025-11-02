# Infrastructure Configuration

This document summarizes Chronos infrastructure components and recommended configurations for production.

## Components

- FastAPI service (HTTP API)
- Celery workers and beat scheduler
- MongoDB (data + GridFS)
- RabbitMQ (broker) and Redis (result backend)
- FIWARE components: Orion, STH‑Comet, IoT Agent
- Observability stack: Loki, Promtail, Grafana (reference Compose)

## Networking & Ports

- API: `GE_PORT` (default 8000)
- RabbitMQ: `5672` (AMQP), `15672` (management UI)
- Redis: `6379`
- MongoDB: `27017`
- Grafana: `3000`, Loki: `3100`

## Environment Configuration

- See `docs/reference/configuration.md` and `docs/admin-guide/configuration.md` for full variables.
- Secrets should be provided via Docker/Kubernetes secrets using `_FILE` suffix where supported.

## Scaling & Concurrency

- API replicas: scale horizontally based on request rate (FastAPI/Uvicorn workers).
- Celery workers: increase `--concurrency` for parallel training; isolate heavy tasks in dedicated nodes if needed.
- Queue separation: keep orchestration separate from data collection and training to prioritize responsiveness.

## Persistence

- MongoDB: replica set for HA; periodic backups using `mongodump`.
- GridFS: consider offloading artifacts to object storage in long‑term retention scenarios.

## Security

- TLS and auth handled at the ingress/proxy layer (Kong, Traefik, NGINX Ingress).
- Network policies to restrict worker access to internal services only.
- RBAC/ServiceAccounts for Kubernetes deployments.

## Observability

- Set `LOG_FORMAT=json` and collect logs with Promtail to Loki; visualize in Grafana.
- Add service dashboards for API latency, task throughput, failure rates, and queue depths.
