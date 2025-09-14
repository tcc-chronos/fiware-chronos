#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Running lint checks with pre-commit"
pre-commit run --all-files
