# syntax=docker/dockerfile:1.7

ARG DISTRO=python:3.11-slim
ARG BUILDER=${DISTRO}
ARG PACKAGE_MANAGER=apt-get

FROM ${BUILDER} AS base
ARG PACKAGE_MANAGER=apt-get

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ARG BUILD_PACKAGES="build-essential curl"
RUN if [ "${PACKAGE_MANAGER}" = "apt-get" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends ${BUILD_PACKAGES} && \
        rm -rf /var/lib/apt/lists/*; \
    elif [ "${PACKAGE_MANAGER}" = "yum" ]; then \
        yum install -y ${BUILD_PACKAGES} && \
        yum clean all; \
    else \
        echo "Unsupported PACKAGE_MANAGER: ${PACKAGE_MANAGER}" && exit 1; \
    fi

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

FROM base AS app-base

COPY . .

ARG GIT_COMMIT=unknown
ARG BUILD_TIME=unknown
ARG VERSION=0.0.0

FROM app-base AS api

ENV GIT_COMMIT=$GIT_COMMIT \
    BUILD_TIME=$BUILD_TIME \
    GE_VERSION=$VERSION \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

LABEL org.opencontainers.image.title="FIWARE Chronos" \
      org.opencontainers.image.description="Time-series forecasting Generic Enabler for the FIWARE ecosystem." \
      org.opencontainers.image.source="https://github.com/tcc-chronos/fiware-chronos" \
      org.opencontainers.image.documentation="https://fiware-chronos.readthedocs.io/en/latest/" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="chronos-maintainers@fiware.org" \
      org.opencontainers.image.vendor="Chronos Working Group" \
      org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.created=$BUILD_TIME \
      org.fiware.contact.localpart="chronos-maintainers" \
      org.fiware.contact.domain="fiware.org"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main.app:app --host 0.0.0.0 --port ${API_PORT:-8000}"]

FROM app-base AS worker

ARG GIT_COMMIT=unknown
ARG BUILD_TIME=unknown
ARG VERSION=0.0.0

ENV GIT_COMMIT=$GIT_COMMIT \
    BUILD_TIME=$BUILD_TIME \
    GE_VERSION=$VERSION \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

LABEL org.opencontainers.image.title="FIWARE Chronos Worker" \
      org.opencontainers.image.description="Background worker image for the FIWARE Chronos Generic Enabler." \
      org.opencontainers.image.source="https://github.com/tcc-chronos/fiware-chronos" \
      org.opencontainers.image.documentation="https://fiware-chronos.readthedocs.io/en/latest/" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="chronos-maintainers@fiware.org" \
      org.opencontainers.image.vendor="Chronos Working Group" \
      org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.created=$BUILD_TIME \
      org.fiware.contact.localpart="chronos-maintainers" \
      org.fiware.contact.domain="fiware.org"

RUN mkdir -p /tmp/models

EXPOSE 5555

CMD ["python", "-m", "src.main"]
