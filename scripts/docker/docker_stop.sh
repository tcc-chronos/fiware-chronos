#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../deploy/docker"

echo "Stopping Fiware-Chronos stack"
docker compose stop
