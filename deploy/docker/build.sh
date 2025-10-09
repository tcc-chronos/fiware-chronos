#!/usr/bin/env bash
set -euo pipefail

REGISTRY=${REGISTRY:-fiware}
IMAGE=${IMAGE:-chronos}
DISTRO=${DISTRO:-python:3.11-slim}
TARGET=${TARGET:-api}

echo "Building ${REGISTRY}/${IMAGE} (target: ${TARGET})"
docker build \
  --build-arg DISTRO="${DISTRO}" \
  --target "${TARGET}" \
  -t "${REGISTRY}/${IMAGE}:local" \
  -f deploy/docker/Dockerfile \
  .
