#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../deploy/docker"

# Load environment variables from .env file
if [ -f "../../.env" ]; then
    set -a  # automatically export all variables
    source ../../.env
    set +a  # disable automatic export
fi

echo "Stopping Fiware-Chronos stack"
docker compose stop
