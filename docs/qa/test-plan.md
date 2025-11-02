# Test Plan

This plan defines the testing strategy for Chronos, including scope, types, environments, tools, and exit criteria.

## Scope

- Validate domain logic, use cases, API contracts, FIWARE gateway interactions, and Celery orchestration.

## Test Types

- Unit tests: domain entities, validators, use cases (pure logic).
- Integration tests: repositories (MongoDB/GridFS), gateways (HTTP clients with mocks).
- Functional tests: end‑to‑end HTTP flows against mocked FIWARE services.
- End‑to‑End smoke: API startup, `/health`, `/info`.
- Static analysis: flake8, mypy; formatting: black, isort.

## Environments

- CI: GitHub Actions with service containers or mocks.
- Local: Docker Compose stack (`deploy/docker/docker-compose.yml`).

## Tools

- pytest, pytest‑cov, respx, httpx, hypothesis (optional),
- flake8, mypy, black, isort,
- mkdocs for docs build validation.

## Execution

```bash
make test          # run pytest with coverage
make lint          # run flake8 + mypy
make format        # run black + isort
mkdocs build       # validate documentation
```

## Coverage

- Threshold: ≥ 90% (line coverage) enforced in CI.
- Reports: terminal summary, XML (`coverage.xml`), and HTML (`htmlcov/`).

## Entry/Exit Criteria

- Entry: features merged behind tests; mocks/stubs available for FIWARE calls.
- Exit: all tests green, coverage ≥ threshold, lint/type checks pass, docs build succeeds.

## Risks & Mitigations

- External FIWARE instability → Use respx to mock HTTP endpoints in CI.
- Long training tasks → Unit test with small synthetic datasets and seeded randoms.

