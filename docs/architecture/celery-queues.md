# Celery Queues (RabbitMQ)

Chronos uses Celery with RabbitMQ as broker and Redis as result backend. The following queues and tasks are configured in `src/infrastructure/services/celery_config.py`.

## Queues and Responsibilities

- data_collection
  - Task: `collect_data_chunk`
  - Purpose: Pull chunks of historical data from STH‑Comet using `hLimit/hOffset`.
  - Notes: Updates `training_jobs` progress; resilient to cancellations.
- model_training
  - Task: `train_model_task`
  - Purpose: Train TensorFlow model, compute metrics, and persist artifacts to GridFS.
  - Notes: Uses preprocessed dataset, respects early stopping and hyperparameters.
- orchestration
  - Task: `orchestrate_training`
  - Purpose: End‑to‑end orchestration: plan collection windows, enqueue chunks, trigger processing and training, finalize status.
- forecast_scheduling
  - Task: `schedule_forecasts`
  - Purpose: Periodically scan `training_jobs` for `prediction_config.enabled` and `next_prediction_at` due; enqueue executions.
  - Trigger: Celery beat schedule `FORECAST_SCHEDULER_INTERVAL_SECONDS` (default 60s).
- forecast_execution
  - Task: `execute_forecast`
  - Purpose: Load artifacts, fetch latest window from STH‑Comet, generate forecast, publish to Orion.

Additional tasks:

- process_collected_data
  - Purpose: Transform collected points into supervised learning tensors; deduce sampling interval.
- cleanup_training_tasks
  - Purpose: Cleanup transient state after training and upon cancellations.

## Topology

```mermaid
flowchart LR
    subgraph RabbitMQ
      DC[data_collection]
      MT[model_training]
      ORC[orchestration]
      FS[forecast_scheduling]
      FE[forecast_execution]
    end
    API[FastAPI] -->|POST /training-jobs| ORC
    ORC --> DC
    DC --> ORC
    ORC --> MT
    FS --> FE
```

## Configuration

- Broker URL: `CELERY_BROKER_URL` (default: `amqp://chronos:chronos@rabbitmq:5672/chronos`).
- Result backend: `CELERY_RESULT_BACKEND` (default: `redis://redis:6379/0`).
- Worker tuning: `worker_prefetch_multiplier=1`, `task_acks_late=true`, `worker_max_tasks_per_child=100`.

