#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Formating code with black and isort"
python -m black .
python -m isort .
