#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:-tccchronos/fiware-chronos}
DOCKER_TARGET=${DOCKER_TARGET:-fiware/chronos}
QUAY_TARGET=${QUAY_TARGET:-quay.io/fiware/chronos}
VERSION=${VERSION:-$(git describe --tags "$(git rev-list --tags --max-count=1)")}

if [[ -z "${VERSION}" ]]; then
  echo "Unable to determine version tag."
  exit 1
fi

echo "Cloning container image ${SOURCE}:${VERSION}"
docker pull -q "${SOURCE}:${VERSION}"

for registry in "$@"; do
  case "${registry}" in
    docker)
      echo "Pushing to Docker Hub: ${DOCKER_TARGET}"
      docker tag "${SOURCE}:${VERSION}" "${DOCKER_TARGET}:${VERSION}"
      docker tag "${SOURCE}:${VERSION}" "${DOCKER_TARGET}:latest"
      docker push -q "${DOCKER_TARGET}:${VERSION}"
      docker push -q "${DOCKER_TARGET}:latest"
      ;;
    quay)
      echo "Pushing to Quay.io: ${QUAY_TARGET}"
      docker tag "${SOURCE}:${VERSION}" "${QUAY_TARGET}:${VERSION}"
      docker tag "${SOURCE}:${VERSION}" "${QUAY_TARGET}:latest"
      docker push -q "${QUAY_TARGET}:${VERSION}"
      docker push -q "${QUAY_TARGET}:latest"
      ;;
    *)
      echo "Unknown registry target '${registry}'"
      ;;
  esac
done
