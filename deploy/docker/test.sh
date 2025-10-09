#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-deploy/docker/docker-compose.yml}

echo "Running container smoke tests using ${COMPOSE_FILE}"
docker compose -f "${COMPOSE_FILE}" up -d api mongo rabbitmq redis

trap 'docker compose -f "${COMPOSE_FILE}" down -v' EXIT

docker compose -f "${COMPOSE_FILE}" ps

echo "Executing health check against Chronos container"
docker compose -f "${COMPOSE_FILE}" exec api curl -sf http://localhost:${GE_PORT:-8000}/health
