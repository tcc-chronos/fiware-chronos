#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../deploy/docker"

echo "Starting Fiware-Chronos stack"
docker compose up "$@"
