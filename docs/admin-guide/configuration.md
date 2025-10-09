# Configuration Reference

Chronos is configured through environment variables or a `.env` file consumed by Pydantic settings. The table below summarises the most relevant options.

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `GE_TITLE` | Human-readable service title. | `chronos-ge` | Displayed in `/info`. |
| `GE_VERSION` | Semantic version of the deployment. | `0.1.0` | Update during releases. |
| `ENVIRONMENT` | Deployment environment name. | `development` | Used for logging context. |
| `DB_MONGO_URI` | MongoDB connection string. | `mongodb://mongo:27017/chronos` | Supports SRV URIs. |
| `DB_DATABASE_NAME` | Mongo database name. | `chronos` | Change to isolate tenants. |
| `CELERY_BROKER_URL` | Broker URL for Celery. | `amqp://chronos:chronos@rabbitmq:5672/chronos` | Support for RabbitMQ/Redis. |
| `CELERY_RESULT_BACKEND` | Backend URL for Celery results. | `redis://redis:6379/0` | Use persistent Redis for resilience. |
| `FIWARE_ORION_URL` | Orion Context Broker endpoint. | – | Required. |
| `FIWARE_STH_URL` | STH-Comet endpoint. | – | Required for training/forecast history. |
| `FIWARE_IOT_AGENT_URL` | IoT Agent endpoint. | – | Required for device provisioning. |
| `FIWARE_SERVICE` | FIWARE service (tenant). | `smart` | Set per deployment. |
| `FIWARE_SERVICE_PATH` | FIWARE service path. | `/` | Supports hierarchical paths. |
| `FORECAST_SCHEDULER_INTERVAL_SECONDS` | Frequency for checking forecast schedules. | `60` | Increase for lower latency. |
| `LOG_LEVEL` | Application log level. | `INFO` | Accepts standard logging levels. |
| `LOG_FORMAT` | Python logging format string. | human-readable | Use JSON for structured logs. |
| `PREDICTION_ATTRIBUTE` | NGSI attribute used to store forecasts. | `forecastSeries` | Align with Smart Data Models naming. |

## Secrets Management

Sensitive fields (e.g., database credentials) support Docker secret indirection via the `_FILE` suffix:

```bash
DB_MONGO_URI_FILE=/run/secrets/mongo_uri
CELERY_BROKER_URL_FILE=/run/secrets/rabbitmq_uri
```

The Docker image entrypoint resolves `_FILE` variants before starting services.

## Logging & Observability

- Set `LOG_FORMAT=json` to emit JSON logs compatible with Loki/ELK.
- Loki configuration is provided in `deploy/docker/promtail-config.yaml`.
- Grafana dashboards can be imported from the `deploy/grafana` directory (to be released).

Refer to the [operations checklist](operations.md) for routine maintenance tasks.
