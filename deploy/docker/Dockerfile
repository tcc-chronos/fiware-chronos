# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

FROM base AS app-base

COPY . .

FROM app-base AS api

ARG GIT_COMMIT=unknown
ARG BUILD_TIME=unknown

ENV GIT_COMMIT=$GIT_COMMIT \
    BUILD_TIME=$BUILD_TIME

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main.app:app --host 0.0.0.0 --port ${API_PORT:-8000}"]

FROM app-base AS worker

RUN mkdir -p /tmp/models

EXPOSE 5555

CMD ["python", "-m", "src.main"]
