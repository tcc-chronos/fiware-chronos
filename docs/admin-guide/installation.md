# Installation Manual

This guide provides step‑by‑step instructions to install and run Chronos in development and production environments.

## Prerequisites

- Python 3.11+ and a POSIX‑like shell (for local dev), or Docker & Docker Compose v2
- Access to FIWARE components (Orion, STH‑Comet, IoT Agent)
- MongoDB, RabbitMQ, and Redis endpoints (or use the provided Compose file)

## Local Development Setup

```bash
git clone https://github.com/tcc-chronos/fiware-chronos.git
cd fiware-chronos
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the correct endpoints for your FIWARE stack and data store. Then run the API:

```bash
make run
# or
python -m src.main.api
```

Open `http://localhost:8000/docs` to explore the Swagger UI.

## Docker Compose (Reference Stack)

The repository includes a full stack under `deploy/docker/`:

```bash
make up ARGS="--build -d"   # API, workers, scheduler, Mongo, Redis, RabbitMQ, Grafana, Loki
docker compose -f deploy/docker/docker-compose.yml ps
curl http://localhost:8000/health
```

Credentials and ports are defined in `.env` and `docker-compose.yml`. See `deploy/docker/README.md` for image build and runtime options.

## Production Recommendations

- Place API behind a reverse proxy (TLS termination, OAuth2/JWT if needed).
- Externalize MongoDB, RabbitMQ, and Redis to managed/HA services.
- Scale Celery workers horizontally; pin CPU/memory for training tasks.
- Enable structured JSON logging and centralize logs (Loki/ELK).
- Backup MongoDB regularly; export artifacts from GridFS for archival.

## Verification

- `GET /health` should report MongoDB, RabbitMQ, Redis connectivity.
- Create a model (`POST /models`) and list it (`GET /models`).
- Start a training job and monitor progress.
