# Testing & QA

Chronos enforces multiple layers of automated testing to guarantee compatibility with FIWARE components.

## Test Types

- **Unit tests** (`tests/unit`) – validate domain entities and use cases using in-memory doubles.
- **Integration tests** (`tests/integration`) – exercise repository implementations and use cases with fake MongoDB backends.
- **Functional tests** (`tests/functional`) – verify HTTP interactions with FIWARE gateways using mocked Orion/STH endpoints.
- **End-to-end tests** (`tests/e2e`) – run the FastAPI application and ensure public endpoints respond correctly.

## Running Tests

```bash
make test          # pytest with coverage >= 90%
make lint          # flake8 + mypy
make format        # black + isort
mkdocs build       # documentation consistency
```

Continuous integration (`.github/workflows/ci.yml`) runs all targets on every pull request.

## Coverage

Coverage thresholds are enforced via `pyproject.toml`. Executing `pytest --cov` generates both terminal and XML reports. HTML reports are stored under `htmlcov/`.

## Functional Orion Test

`tests/functional/test_orion_integration.py` spins up mocked Orion endpoints using `respx` to ensure the gateway:

- Builds correct FIWARE headers.
- Handles entity existence checks and creation.
- Publishes forecasts via `/v2/entities/{id}/attrs`.
- Interprets subscription IDs returned in the `Location` header.

This safeguards compatibility with the FIWARE Context Broker without requiring a live Orion instance during CI.

## Adding New Tests

- Place domain-centric tests under `tests/unit`.
- Add new integration tests alongside the infrastructure component under test.
- Update coverage omit rules in `pyproject.toml` if new files should be excluded.
- Document new testing strategies here and in the README.
