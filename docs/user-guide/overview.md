# User Guide Overview

Chronos simplifies forecasting within FIWARE-based ecosystems. This guide introduces the core concepts and provides the minimum steps to ingest data, train models, and publish predictions.

## Prerequisites

- Running FIWARE stack including Orion Context Broker, IoT Agent, and STH-Comet.
- Chronos deployed locally or via Docker Compose (`deploy/docker/docker-compose.yml`).
- API credentials to access your FIWARE services (tenant/service path).

## Core Concepts

- Model - Configuration describing the neural architecture (LSTM/GRU), FIWARE entity metadata, and training hyperparameters.
- Training Job - Asynchronous process that fetches history from STH-Comet, trains TensorFlow models, and stores artifacts in MongoDB/GridFS.
- Prediction - Forecast results published as NGSI attributes in Orion and optionally subscribed to STH-Comet.

## Quickstart

1. **Create a model**

```bash
curl -X POST http://localhost:8000/models \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "smart-building-humidity",
    "entity_id": "urn:ngsi-ld:Sensor:001",
    "entity_type": "Sensor",
    "feature": "humidity",
    "model_type": "LSTM",
    "lookback_window": 48,
    "forecast_horizon": 6
  }'
```
2. **Trigger training**

```bash
curl -X POST http://localhost:8000/models/{model_id}/training-jobs
```

3. **Check status**

```bash
curl http://localhost:8000/models/{model_id}/training-jobs/{job_id}
```

4. **Publish forecasts**

```bash
curl -X POST http://localhost:8000/models/{model_id}/training-jobs/{job_id}/prediction-toggle \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true}'
```

Chronos creates the target entity in Orion when absent and updates the attribute `forecastSeries` by default. Adjust attribute names through the API payload.

## Next Steps

- Explore the [API walkthrough](api-walkthrough.md) for additional examples.
- Review the [Deployment Guide](../admin-guide/deployment.md) for production hardening.
- Monitor pipelines via the Grafana dashboards described in the [Operations Checklist](../admin-guide/operations.md).
