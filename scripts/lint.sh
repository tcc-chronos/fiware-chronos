#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Running format checks first to ensure consistent code"
./scripts/format.sh >/dev/null 2>&1 || true

echo "Running lint checks with pre-commit"
pre-commit run --all-files
