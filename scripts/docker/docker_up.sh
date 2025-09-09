#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../deploy/docker"

export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
export BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "Starting Fiware-Chronos stack"
docker compose up "$@"
