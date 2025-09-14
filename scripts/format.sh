#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Formatting code using pre-commit hooks"
pre-commit run black --all-files
pre-commit run isort --all-files
