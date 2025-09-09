#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "Running API on port ${API_PORT:-8000}..."
python -m uvicorn apps.api.main:app --reload --host 0.0.0.0 --port "${API_PORT:-8000}"
