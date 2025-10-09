# FIWARE Chronos Container Image

Chronos ships an official container image published to Docker Hub (`fiware/chronos`) and Quay.io (`quay.io/fiware/chronos`). This guide documents build options, environment variables, secrets, and runtime examples.

## Image Tags

- `latest` – development snapshot.
- `<semver>` – released versions (e.g., `1.0.0`).
- `FIWARE_<major>.<minor>` – aligned with FIWARE catalogue releases (e.g., `FIWARE_1.0`).

## Multi-Stage Build

The reference `deploy/docker/Dockerfile` supports multiple stages:

- `base` – installs system dependencies and Python requirements.
- `api` – runs the FastAPI application.
- `worker` – runs Celery worker/beat processes.

### Build Command

```bash
docker build \
  --build-arg DISTRO=python:3.11-slim \
  --target api \
  -t fiware/chronos:local \
  -f deploy/docker/Dockerfile \
  .
```

Set `DISTRO` to alternate base images (e.g., Red Hat UBI) when required.

## Environment Variables

The container supports the same variables listed in the [Configuration Reference](../../docs/admin-guide/configuration.md). Key variables include:

| Variable | Description |
|----------|-------------|
| `GE_TITLE`, `GE_VERSION` | Metadata exposed on `/info`. |
| `DB_MONGO_URI`, `DB_DATABASE_NAME` | MongoDB connection parameters. |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | RabbitMQ/Redis endpoints. |
| `FIWARE_ORION_URL`, `FIWARE_STH_URL`, `FIWARE_IOT_AGENT_URL` | FIWARE component endpoints. |
| `FIWARE_SERVICE`, `FIWARE_SERVICE_PATH` | FIWARE headers used for Orion/STH calls. |
| `PREDICTION_ATTRIBUTE` | Attribute name used to publish forecasts. |

### Docker Secrets

Sensitive values support `_FILE` indirection:

```bash
docker run \
  --env-file .env \
  -e DB_MONGO_URI_FILE=/run/secrets/mongo_uri \
  -e CELERY_BROKER_URL_FILE=/run/secrets/rabbit_uri \
  --secret source=mongo_uri,target=mongo_uri \
  --secret source=rabbit_uri,target=rabbit_uri \
  fiware/chronos:1.0.0
```

## Docker Compose

`deploy/docker/docker-compose.yml` provides a full FIWARE-ready stack. To launch:

```bash
docker compose -f deploy/docker/docker-compose.yml up -d
```

To run a minimal smoke test (API + dependencies):

```bash
./deploy/docker/test.sh
```

## Exposed Ports

- `API_PORT` (default `8000`) – FastAPI HTTP interface.
- `5555` – optional Celery Flower/worker telemetry.

## OCI Labels

The image embeds OCI metadata (authors, description, documentation URL, license, revision, vendor) to simplify auditing. Inspect using:

```bash
docker inspect --format '{{json .Config.Labels}}' fiware/chronos:1.0.0 | jq
```
