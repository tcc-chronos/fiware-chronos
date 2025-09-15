#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Formatting code using pre-commit hooks"
echo "Running black formatter..."
pre-commit run black --all-files
echo "Running isort formatter..."
pre-commit run isort --all-files

echo "Format checks completed. Files are now properly formatted."
