# Architecture Overview

Chronos adheres to Clean Architecture: domain‑centric, with infrastructure and presentation details at the edges. Long‑running operations are delegated to Celery workers.

## Layered View

```mermaid
flowchart LR
  subgraph Presentation
    API[FastAPI Routers]
  end
  subgraph Application
    UC[Use Cases & DTOs]
  end
  subgraph Domain
    ENT[Entities & Ports]
  end
  subgraph Infrastructure
    DB[MongoDatabase/GridFS]
    GW[FIWARE Gateways]
    CEL[Celery Tasks]
  end

  API --> UC
  UC --> ENT
  UC --> DB
  UC --> GW
  UC --> CEL
  CEL --> DB
  CEL --> GW
  GW -->|NGSI-v2| ORION[(Orion)]
  GW --> STH[(STH-Comet)]
  GW --> IOTA[(IoT Agent)]
  DB --> MONGO[(MongoDB)]
```

## Key Components

- Presentation: FastAPI controllers under `src/presentation/controllers`.
- Application: Coordinating use cases in `src/application/use_cases`.
- Domain: Entities and interfaces in `src/domain`.
- Infrastructure: MongoDB access, GridFS, FIWARE gateways, Celery tasks.

## Execution Flows

See `sequence-diagrams.md` for detailed training, prediction, and scheduling flows.
