# Environment Variables

The following table summarises all environment variables consumed by Chronos. Values marked **required** must be set for production deployments.

| Variable | Required | Description |
|----------|----------|-------------|
| `GE_TITLE` | No | Display name returned by `/info`. |
| `GE_DESCRIPTION` | No | Human readable description in `/info`. |
| `GE_VERSION` | Yes | SemVer string for the running build. |
| `GE_PORT` | No | HTTP port exposed by the API (default `8000`). |
| `ENVIRONMENT` | No | Environment label (`development`, `staging`, `production`). |
| `DB_MONGO_URI` | Yes | MongoDB connection URI. Supports credentials and SRV records. |
| `DB_DATABASE_NAME` | Yes | MongoDB database name used for models and jobs. |
| `CELERY_BROKER_URL` | Yes | Celery broker connection string (RabbitMQ/Redis). |
| `CELERY_RESULT_BACKEND` | Yes | Celery result backend connection string. |
| `FIWARE_ORION_URL` | Yes | Base URL of the Orion Context Broker instance. |
| `FIWARE_STH_URL` | Yes | Base URL of the STH-Comet instance. |
| `FIWARE_IOT_AGENT_URL` | Yes | Base URL of the IoT Agent instance. |
| `FIWARE_SERVICE` | Yes | FIWARE service (tenant). |
| `FIWARE_SERVICE_PATH` | Yes | FIWARE service path (`/` for root). |
| `FIWARE_STH_MAX_PER_REQUEST` | No | Number of records fetched per STH request (default `100`). |
| `FORECAST_SCHEDULER_INTERVAL_SECONDS` | No | Frequency (in seconds) for Celery beat to evaluate forecast schedules. |
| `LOG_LEVEL` | No | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FORMAT` | No | Logging format string. Use `"json"` for JSON logs. |
| `PREDICTION_ATTRIBUTE` | No | NGSI attribute used for predictions (`forecastSeries`). |
| `TRAINING_PARALLELISM` | No | Max number of parallel training tasks. |
| `TENANT_DEFAULT_LOOKBACK` | No | Default lookback window for new models. |

Environment variables can be supplied via:

- `.env` file (local development)
- Docker Compose `env_file`
- Kubernetes secrets/config maps
- Docker secrets using `_FILE` suffix (e.g., `DB_MONGO_URI_FILE`)

See the [Configuration Guide](../admin-guide/configuration.md) for operational guidance.
