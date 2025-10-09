# API Walkthrough

This walkthrough demonstrates the main Chronos APIs using `curl`. All requests assume Chronos runs locally on `http://localhost:8000` and the FIWARE tenant/service path is configured in `.env`.

## List Models

```bash
curl http://localhost:8000/models
```

Expected response (abbreviated):

```json
[
  {
    "id": "f71c0cfa-8c4f-4b61-9e48-0d45d4f8e3af",
    "name": "smart-building-humidity",
    "status": "trained",
    "entity_id": "urn:ngsi-ld:Sensor:001",
    "feature": "humidity"
  }
]
```

## Create Training Job

```bash
curl -X POST http://localhost:8000/models/{model_id}/training-jobs \
  -H 'Content-Type: application/json' \
  -d '{"last_n": 1000}'
```

Chronos enqueues an asynchronous task, returning the training job identifier.

## Monitor Training

```bash
curl http://localhost:8000/models/{model_id}/training-jobs/{job_id}
```

Key fields include:

- `status`: `pending`, `collecting_data`, `training`, `completed`, or `failed`.
- `metrics`: MAE, RMSE, R², etc.
- `artifacts`: object IDs stored in GridFS.

## On-Demand Predictions

```bash
curl -X POST http://localhost:8000/models/{model_id}/training-jobs/{job_id}/predict
```

Response snippet:

```json
{
  "context_window": [
    {"timestamp": "2025-01-04T10:00:00Z", "value": 61.3},
    {"timestamp": "2025-01-04T10:05:00Z", "value": 60.9}
  ],
  "forecast_horizon": [
    {"timestamp": "2025-01-04T10:35:00Z", "value": 63.1},
    {"timestamp": "2025-01-04T10:40:00Z", "value": 63.7}
  ]
}
```

## Enable Recurring Forecasts

```bash
curl -X POST http://localhost:8000/models/{model_id}/training-jobs/{job_id}/prediction-toggle \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true, "sampling_interval_seconds": 900}'
```

The scheduler triggers `execute_forecast` periodically and publishes results to Orion.

## Health & Info Endpoints

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
```

- `/health` aggregates database, queue, and scheduler status.
- `/info` returns build metadata, commit hash, and FIWARE dependencies.

Refer to the [OpenAPI UI](http://localhost:8000/docs) for schema details. The static schema is also served at `/openapi.json` and mirrored on Read the Docs in the [API Reference](../reference/api.md).
