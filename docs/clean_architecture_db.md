# Clean Architecture – MongoDB Responsibility Split

This document outlines how Chronos separates concerns between repositories and low-level database access, following Clean Architecture principles.

## Implemented Structure

### 1. `MongoDatabase` (`src/infrastructure/database/mongo_database.py`)

Responsibilities:

- Manage MongoDB connections and database lifecycle.
- Expose direct collection accessors.
- Provide low-level CRUD helpers.
- Create indexes on start-up.
- Remain agnostic of domain entities (operates on BSON dictionaries).

### 2. `ModelRepository` Implementation (`src/infrastructure/repositories/model_repository.py`)

Responsibilities:

- Depend on the `MongoDatabase` abstraction.
- Convert between domain entities and MongoDB documents.
- Apply domain-specific validation and error handling (`ModelNotFoundError`, `ModelOperationError`).
- Define collection-specific indexes.

## Benefits

1. **SOLID Alignment**
   - Single Responsibility: Infrastructure and domain rules are isolated.
   - Dependency Inversion: Application layer depends on interfaces, not MongoDB details.
2. **Improved Testability**
   - `MongoDatabase` can be mocked or replaced with `FakeMongoDatabase` during tests.
   - Repository tests validate mapping logic without touching real infrastructure.
3. **Reuse**
   - Shared connection logic lives in `MongoDatabase` and is reused by other repositories.
4. **Maintainability**
   - Infrastructure changes (e.g., connection pooling) are contained.
   - Domain rule changes do not leak into the database layer.
5. **Clear Boundary**
   - Domain → Repository interface → Infrastructure implementation.

## Dependency Injection Setup

```python
mongo_database = providers.Singleton(
    MongoDatabase,
    mongo_uri=config.database.mongo_uri,
    db_name=config.database.database_name,
)

model_repository = providers.Singleton(
    ModelRepository,
    database=mongo_database,
)
```

## Extensibility

To add a new repository:

1. Define an interface in `src/domain/repositories`.
2. Implement the interface in `src/infrastructure/repositories`.
3. Register the implementation in the DI container (`src/main/container.py`).

This keeps new persistence logic consistent with existing components and prevents infrastructure leakage into the domain.
