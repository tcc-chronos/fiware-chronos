# Deployment Guide

This document describes how to deploy Chronos in production environments and integrate it with FIWARE components.

## Architecture Overview

Chronos is composed of:

- **API service (FastAPI)** handling public endpoints.
- **Celery workers** running training and forecasting jobs.
- **Celery beat scheduler** orchestrating recurring forecasts.
- **MongoDB** storing configurations, training jobs, and artefacts (GridFS).
- **RabbitMQ / Redis** for task orchestration and result backend.
- **Grafana, Loki, Promtail** for observability.

The reference deployment uses Docker Compose (`deploy/docker/docker-compose.yml`) and can be adapted to Kubernetes or bare metal.

## Container Images

- Official image: `fiware/chronos:<version>`
- Tags:
  - `latest` – current development snapshot.
  - `<semver>` – released versions (e.g., `1.0.0`).
  - `FIWARE_<major>.<minor>` – FIWARE catalogue aligned releases (e.g., `FIWARE_1.0`).
- Images are multi-arch (amd64/arm64). Verify the [container README](https://github.com/tcc-chronos/fiware-chronos/blob/main/deploy/docker/README.md) for build arguments and supported distributions.

## Deployment Steps

1. **Prepare Environment**
   - Provide `.env` file using `.env.example` as a template.
   - Secure secrets via Docker secrets or environment variable files (see [Configuration](configuration.md)).
2. **Launch Stack**
   ```bash
   make up ARGS="--build -d"
   ```
3. **Verify Services**
   - `docker compose ps`
   - `curl http://localhost:8000/health`
4. **Provision FIWARE Context**
   - Ensure Orion, IoT Agent, and STH-Comet are reachable from Chronos.
   - Configure service/group and device metadata as needed.

## Scaling Guidance

- Deploy MongoDB and RabbitMQ as managed services for high availability.
- Run multiple worker replicas to parallelise training jobs (`docker compose up --scale worker=3`).
- Use Kubernetes Horizontal Pod Autoscalers for API and workers based on CPU usage.
- Configure Prometheus + Grafana dashboards for long-term metrics retention.

## Security Considerations

- Enforce TLS offloading (e.g., reverse proxy with cert-manager).
- Rotate secrets regularly and use `_FILE` suffixed environment variables for Docker secrets.
- Restrict network ingress to the API tier; workers should not be publicly accessible.
- Configure Role Based Access Control (RBAC) when deploying on Kubernetes.

## Backup & Disaster Recovery

- Schedule MongoDB dumps (e.g., `mongodump`) and store them securely.
- Export trained artefacts from GridFS to object storage for long-term retention.
- Document restore procedures in runbooks stored alongside this guide.
