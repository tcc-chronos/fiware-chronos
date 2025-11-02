# Use Cases and Business Rules

This document describes primary use cases and the business rules governing them.

## Use Case Model

```mermaid
%%{init: { 'theme': 'neutral' }}%%
usecaseDiagram
actor Admin as "Admin/Operator"
actor User as "API Client"

rectangle Chronos {
  (UC1: Manage Models)
  (UC2: Launch Training)
  (UC3: Monitor Training)
  (UC4: On-demand Prediction)
  (UC5: Enable Recurring Forecasts)
  (UC6: Publish to Orion)
  (UC7: Discover Devices)
  (UC8: Health & Info)
}

User --> (UC1: Manage Models)
User --> (UC2: Launch Training)
User --> (UC3: Monitor Training)
User --> (UC4: On-demand Prediction)
Admin --> (UC5: Enable Recurring Forecasts)
Admin --> (UC7: Discover Devices)
User --> (UC8: Health & Info)
(UC2: Launch Training) ..> (UC6: Publish to Orion) : includes
(UC4: On-demand Prediction) ..> (UC6: Publish to Orion) : includes
```

## UC1: Manage Models

- Description: Create, update, list, retrieve, and delete model definitions.
- Actors: API Client
- Pre-conditions: MongoDB reachable.
- Post-conditions: Model stored with status `draft` or `trained`.
- Main Flow:
  - POST `/models` with hyperparameters and FIWARE metadata.
  - GET `/models` or `/models/{id}` to verify.
- Business Rules:
  - Required: `entity_type`, `entity_id`, `feature`, `lookback_window`, `forecast_horizon`.
  - Model types limited to `lstm` or `gru`.

## UC2: Launch Training

- Description: Start a training job that collects data and trains a model.
- Actors: API Client
- Pre-conditions: Model exists; STH-Comet reachable.
- Post-conditions: Training job created; artifacts persisted; metrics recorded.
- Main Flow:
  - POST `/models/{id}/training-jobs`
  - Celery tasks: `collect_data_chunk` -> `process_collected_data` -> `train_model_task` -> `cleanup_training_tasks`.
- Business Rules:
  - Training can be cancelled; status transitions preserved.
  - Data collection uses hLimit/hOffset windows; progress tracked.

## UC3: Monitor Training

- Description: Track progress, timings, and errors of a training job.
- Actors: API Client
- Pre-conditions: Training job exists.
- Post-conditions: None.
- Main Flow:
  - GET `/models/{id}/training-jobs/{job_id}`
- Business Rules:
  - Expose `status`, `metrics`, `start/end` times, and `error` details.

## UC4: On-demand Prediction

- Description: Generate forecast for latest context data.
- Actors: API Client
- Pre-conditions: Training completed with artifacts.
- Post-conditions: Forecast published to Orion and optionally returned.
- Main Flow:
  - POST `/models/{id}/training-jobs/{job_id}/predict`
  - Load artifacts from GridFS; fetch window from STH-Comet; infer; publish.
- Business Rules:
  - Forecast horizon equals training configuration unless overridden by policy.

## UC5: Enable Recurring Forecasts

- Description: Configure automatic, periodic predictions.
- Actors: Admin/Operator
- Pre-conditions: Training completed; Orion and IoT Agent reachable.
- Post-conditions: Next execution timestamp scheduled; beat evaluates.
- Main Flow:
  - POST `/models/{id}/training-jobs/{job_id}/prediction-toggle` with `enabled=true`.
  - Celery beat runs `schedule_forecasts`; worker executes `execute_forecast`.
- Business Rules:
  - Creates/ensures service groups, devices, Orion entity, and subscription.

## UC6: Publish to Orion

- Description: Upsert predictions into Orion as NGSI attributes and manage subscriptions.
- Actors: System
- Business Rules:
  - Respect FIWARE tenant headers (`Fiware-Service`, `Fiware-ServicePath`).
  - Attribute naming controlled by `PREDICTION_ATTRIBUTE` env var.

## UC7: Discover Devices

- Description: Fetch IoT devices and available attributes via IoT Agent.
- Actors: Admin/Operator
- Main Flow:
  - GET `/devices?service=&service_path=`

## UC8: Health & Info

- Description: Report system health and build information.
- Actors: Any
- Main Flow:
  - GET `/health`, `/info`

## Cross-cutting Business Rules

- Robust error handling with contextual logging.
- Idempotent operations where feasible (entity/attribute upserts).
- Coverage target >= 90%; changes must include tests.

