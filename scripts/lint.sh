#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Running flake8 and mypy"
python -m flake8 .
python -m mypy .
