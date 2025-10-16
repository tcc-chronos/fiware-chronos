#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../deploy/docker"

# Load environment variables from .env file
if [ -f "../../.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source ../../.env
    set +a  # disable automatic export
else
    echo "Warning: .env file not found. Using default values."
fi

export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
export BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export COMPOSE_DOCKER_CLI_BUILD=${COMPOSE_DOCKER_CLI_BUILD:-1}

if [ -z "${COMPOSE_PROFILES+x}" ]; then
    export COMPOSE_PROFILES="observability"
fi

echo "Starting Fiware-Chronos stack"
docker compose up "$@"
