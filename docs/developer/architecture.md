# Architecture Overview (Developer)

Chronos applies Clean Architecture to keep business logic independent from infrastructure concerns. This page focuses on code layout and wiring from a developer’s perspective.

## Layered Structure

- Domain (`src/domain`) - Entities, value objects, and repository/gateway interfaces. Pure Python with no framework dependencies.
- Application (`src/application`) - Use cases orchestrating domain logic, FIWARE gateways, and persistence adapters.
- Infrastructure (`src/infrastructure`) - Implementations for MongoDB, Celery, Orion/STH/IoT Agent gateways, and external services.
- Presentation (`src/presentation`) - FastAPI routers and request/response DTOs.
- Main (`src/main`) - Dependency injection container (`dependency_injector`) wiring and service bootstrap.

## Data Flow

1. API request enters a FastAPI router.
2. Router delegates to a use case via dependency injection.
3. Use case calls repository and gateway abstractions.
4. Infrastructure adapters translate to concrete clients (MongoDB, httpx, Celery).
5. Responses are serialized back through DTOs.

## Key Integrations

- OrionGateway - Publishes predictions (`/v2/entities/{id}/attrs`) and creates subscriptions.
- STHCometGateway - Fetches time series data via `/STH/v1/contextEntities`.
- IoTAgentGateway - Provisions devices/service groups used during forecast publication.
- Celery - Orchestrates background jobs for training and scheduled forecasts.

See `../clean_architecture_db.md` for repository patterns and `../logging.md` for observability design.
