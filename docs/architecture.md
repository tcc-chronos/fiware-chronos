# FIWARE Chronos — Architecture Overview (Clean)

This page provides a concise overview of Chronos’ architecture and points to detailed diagrams and flows. It avoids duplication with other pages to keep the documentation cohesive and easy to maintain.

## Overview

Chronos orchestrates time‑series model training and forecasting integrated with FIWARE. The FastAPI service handles HTTP requests while long‑running pipelines execute on Celery workers. MongoDB (with GridFS) stores state and artifacts; RabbitMQ and Redis back Celery for queuing and results.

## Clean Architecture

Chronos follows Clean Architecture with clear boundaries:

- Presentation: FastAPI routers and DTOs (`src/presentation`).
- Application: Use cases coordinating domain, gateways, and repositories (`src/application`).
- Domain: Entities and ports with business rules (`src/domain`).
- Infrastructure: MongoDB/GridFS repositories, FIWARE gateways, Celery tasks (`src/infrastructure`).

See “Architecture Diagrams” for a visual layer diagram: architecture/overview.md

## Key Execution Flows

Detailed sequence diagrams are available on “Sequence Diagrams”: architecture/sequence-diagrams.md. They cover:

- End‑to‑end training (orchestration, data collection, preprocessing, training, artifact persistence).
- On‑demand prediction (artifact load, recent window fetch, inference, Orion publish).
- Recurring predictions (toggle, beat scheduler, execution and publish).

## Messaging and Background Processing

Asynchronous processing is implemented with Celery. Queues and task responsibilities are documented in “Celery Queues”: architecture/celery-queues.md

## FIWARE Integrations

- IoT Agent: device discovery and provisioning of service groups/devices when enabling forecasts.
- Orion Context Broker: ensure entities/subscriptions and upsert predictions as NGSI attributes.
- STH‑Comet: historical data collection and prediction history retrieval.

## Endpoints

The REST API surface is documented in “API Reference”: reference/api.md
