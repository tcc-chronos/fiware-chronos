# Architecture Overview

Chronos applies Clean Architecture principles to keep business logic independent from infrastructure concerns. This section summarises the main layers and how they interact.

## Layered Structure

- **Domain** (`src/domain`) – Entities, value objects, and repository/gateway interfaces. Pure Python with no framework dependencies.
- **Application** (`src/application`) – Use cases orchestrating domain logic, FIWARE gateways, and persistence adapters.
- **Infrastructure** (`src/infrastructure`) – Implementations for MongoDB, Celery, Orion/STH/IoT Agent gateways, and external services.
- **Presentation** (`src/presentation`) – FastAPI routers and request/response DTOs.
- **Main** (`src/main`) – Dependency injection container (`dependency_injector`) wiring and service bootstrap.

## Data Flow

1. API request enters FastAPI router.
2. Router delegates to a use case via dependency injection.
3. Use case calls repositories/gateways abstractions.
4. Infrastructure adapters translate to concrete clients (MongoDB, httpx, Celery).
5. Responses are serialised back through DTOs.

## Key Integrations

- **OrionGateway** – publishes predictions (`/v2/entities/{id}/attrs`) and creates subscriptions.
- **STHCometGateway** – fetches time series data via `/STH/v1/contextEntities`.
- **IoTAgentGateway** – provisions devices/service groups used during forecast publication.
- **Celery** – orchestrates background jobs for training and scheduled forecasts.

See [`clean_architecture_db.md`](../clean_architecture_db.md) for repository patterns and [`logging.md`](../logging.md) for observability design.
