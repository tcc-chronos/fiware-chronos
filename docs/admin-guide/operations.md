# Operations Checklist

Use this checklist to maintain a healthy Chronos deployment.

## Daily

- Review `make health` or `/health` endpoint for dependency status.
- Monitor Celery queues (`orchestration`, `data_collection`, `forecast_execution`) through the RabbitMQ Management UI.
- Inspect Grafana dashboards for training durations, queue depth, and forecast latency.

## Weekly

- Rotate Celery worker logs and archive them to long-term storage.
- Validate that scheduled forecasts execute on time and Orion entities are updated.
- Review error rates in Loki for recurrent exceptions (e.g., STH connectivity).
- Execute `make test` against the staging environment to ensure reproducibility.

## Monthly

- Refresh Docker images with the latest base image patches (`make docker-build`).
- Rebuild machine learning models if significant concept drift is detected.
- Audit user access to infrastructure (RabbitMQ, MongoDB, Grafana).
- Submit QA results to the FIWARE QA Lab and update the QA rating in the README if necessary.

## Incident Response

- **Training failure** – Inspect Celery worker logs. Re-run job via `POST /models/{id}/training-jobs`.
- **Forecast publish failure** – Check Orion connectivity and ensure FIWARE headers are correct. Use the functional test in `tests/functional/test_orion_integration.py` as a diagnostic.
- **Data ingestion issues** – Verify STH-Comet availability and that subscriptions are active.

Document any incidents and mitigation steps within your organisational runbooks. Significant incidents affecting FIWARE compatibility should be reported to the FIWARE Help Desk.
