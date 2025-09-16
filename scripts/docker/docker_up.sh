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

echo "Starting Fiware-Chronos stack"
docker compose up "$@"
