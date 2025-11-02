# FIWARE Chronos

[![FIWARE Chapter](https://img.shields.io/badge/FIWARE-Processing/Analysis-88a1ce.svg)](https://github.com/FIWARE/catalogue/tree/master/processing)
[![FIWARE Generic Enabler](https://img.shields.io/badge/FIWARE-Generic_Enabler-0b7fab.svg)](https://www.fiware.org/developers/catalogue/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/TBD/badge)](https://bestpractices.coreinfrastructure.org/projects/TBD)
[![Documentation](https://readthedocs.org/projects/fiware-chronos/badge/?version=latest)](https://fiware-chronos.readthedocs.io/en/latest/)
[![CI](https://github.com/tcc-chronos/fiware-chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/tcc-chronos/fiware-chronos/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/fiware/chronos.svg)](https://hub.docker.com/r/fiware/chronos)

> **Elevator pitch:** Chronos delivers automated time-series forecasting for FIWARE ecosystems, orchestrating data ingestion, model training, and prediction publication to the Orion Context Broker with zero-touch lifecycle management.

This project is part of [FIWARE](https://www.fiware.org/). For more information check the FIWARE Catalogue entry for
[Processing & Analysis](https://github.com/Fiware/catalogue/tree/master/processing).

| [Documentation](https://fiware-chronos.readthedocs.io/en/latest/) | [FIWARE Academy](https://fiware-academy.readthedocs.io/en/latest/) | [Smart Data Models](https://smartdatamodels.org/) | [FIWARE Helpdesk](https://helpdesk.fiware.org/) |

## Contents

- [About Chronos](#about-chronos)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Docker Deployment](#docker-deployment)
- [Usage Walkthrough](#usage-walkthrough)
- [Testing & Quality](#testing--quality)
- [Support & Community](#support--community)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Releases](#releases)
- [License](#license)

## About Chronos

Chronos is a FIWARE Generic Enabler that manages the full lifecycle of deep learning forecasting models for IoT sensor data. It integrates natively with **Orion Context Broker**, **IoT Agent**, and **STH-Comet** to:

- Collect historical data and train recurrent neural networks (LSTM/GRU).
- Schedule asynchronous training pipelines with Celery workers.
- Publish forecasts as NGSI attributes and optional subscriptions for downstream analytics.
- Provide observability via Loki, Promtail, and Grafana dashboards.

Chronos fits the **Processing & Analysis** FIWARE chapter and targets solutions that need adaptive forecasting tightly coupled with NGSI context management.

## Architecture

![Chronos Architecture](https://github.com/user-attachments/assets/ef896d44-32df-437f-bf9a-262290983d4a)

Chronos follows Clean Architecture principles:

- **Domain** – entity definitions (`Model`, `TrainingJob`, `PredictionRecord`) and invariants.
- **Application** – use cases orchestrating FIWARE gateways, repositories, and schedulers.
- **Infrastructure** – adapters for MongoDB/GridFS, Orion, STH-Comet, IoT Agent, Celery.
- **Presentation** – FastAPI controllers exposing management and forecasting APIs.
- **Workers** – Celery beat/worker processes for background orchestration.

Detailed diagrams and entity relationships are available in [`docs/clean_architecture_db.md`](docs/clean_architecture_db.md) and the [User Guide](https://fiware-chronos.readthedocs.io/en/latest/user-guide/overview/).

## Documentation

Comprehensive documentation is published on Read the Docs:

- [User Guide](https://fiware-chronos.readthedocs.io/en/latest/user-guide/overview/)
- [Installation & Administration Guide](https://fiware-chronos.readthedocs.io/en/latest/admin-guide/deployment/)
- [API Reference](https://fiware-chronos.readthedocs.io/en/latest/reference/api/)
- [Developer Topics](https://fiware-chronos.readthedocs.io/en/latest/developer/architecture/)

### Technical Deliverables

The repository includes a complete documentation set aligned with software engineering best practices:

- Requirements Specification: `docs/specs/requirements.md`
- Use Cases & Business Rules: `docs/specs/use-cases.md`
- Database Modeling: `docs/database/modeling.md`
- Celery/RabbitMQ Queues: `docs/architecture/celery-queues.md`
- Architecture Diagrams: `docs/architecture/overview.md`
- Class Diagram: `docs/architecture/class-diagram.md`
- Sequence Diagrams: `docs/architecture/sequence-diagrams.md`
- API Reference: `docs/reference/api.md`
- Installation Manual: `docs/admin-guide/installation.md`
- Infrastructure Configuration: `docs/infrastructure/configuration.md`
- Test Plan: `docs/qa/test-plan.md`
- Test Report Template: `docs/qa/test-report.md`

All documents are organized for publication via MkDocs. See `mkdocs.yml` for the navigation tree.

Documentation sources are stored under `docs/` and built via MkDocs with FIWARE Read the Docs CSS (see [`mkdocs.yml`](mkdocs.yml)).

## Getting Started

### Prerequisites

- Linux host with AVX-capable CPU (for TensorFlow)
- Docker & Docker Compose v2
- Python 3.12
- FIWARE stack (Orion, IoT Agent, STH-Comet) reachable from Chronos

### Local Installation

```bash
git clone https://github.com/tcc-chronos/fiware-chronos.git
cd fiware-chronos
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
make run
```

The `.env` file documents all configuration parameters. Refer to the Installation guide for production hardening, TLS, and scaling advice.

### Docker Deployment

Chronos ships a reference Docker Compose stack under `deploy/docker`:

```bash
make up ARGS="--build -d"   # Build and launch API, workers, scheduler, Mongo, Redis, RabbitMQ, Grafana, Loki
make stop                   # Stop running containers
```

- API – `http://localhost:${GE_PORT:-8000}`
- RabbitMQ Management – `http://localhost:15672` (`chronos/chronos`)
- Grafana – `http://localhost:3000` (`admin/admin`)

For container image documentation, see [`deploy/docker/README.md`](deploy/docker/README.md). Publishing to Docker Hub / Quay.io is automated via `.github/fiware/image-clone.sh`.

## Usage Walkthrough

1. **Register a model** – `POST /models` with FIWARE metadata (entity ID/type, observed attribute) and training hyperparameters.
2. **Launch training** – `POST /models/{id}/training-jobs`, which collects history from STH-Comet, trains TensorFlow models, stores artifacts in GridFS, and persists metrics.
3. **Request forecasts** – `POST /models/{id}/training-jobs/{job}/predict` for on-demand predictions or enable recurring forecasts via `prediction-toggle`.
4. **Publish to Orion** – predictions are upserted as NGSI attributes with optional subscription creation so STH-Comet can ingest generated series.

API examples (with `curl`) are provided in the [API Walkthrough](https://fiware-chronos.readthedocs.io/en/latest/user-guide/api-walkthrough/).

## Testing & Quality

- **Linting & typing:** `make lint` (flake8 + mypy) must pass before merging.
- **Formatting:** `make format` (black + isort).
- **Unit & integration tests:** `make test` runs pytest with coverage (threshold ≥ 90%).
- **Functional FIWARE tests:** see [`tests/functional/test_orion_integration.py`](tests/functional/test_orion_integration.py) for NGSI interactions.
- **CI:** GitHub Actions workflow [`ci.yml`](.github/workflows/ci.yml) executes lint, type check, unit/integration tests, and documentation build per pull request.

Badges above reflect the current status of these quality gates.

| QA Metric | Status |
|-----------|--------|
| FIWARE QA Rating | Pending (target: A) |

## Support & Community

- **Issue tracking:** GitHub Issues – please label bugs, enhancements, and questions appropriately.
- **Helpdesk:** Engage with the FIWARE QA team via [helpdesk.fiware.org](https://helpdesk.fiware.org/).
- **Stack Overflow:** use the [`fiware-chronos`](https://stackoverflow.com/questions/tagged/fiware-chronos) tag for technical questions. Questions will receive triage within two business days.
- **Security reports:** Follow the responsible disclosure policy outlined in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Known issues and troubleshooting guidance are documented in the Installation & Administration Guide.

## Roadmap

The product roadmap follows the FIWARE template and is published in [`ROADMAP.md`](ROADMAP.md). Planned milestones are aligned with FIWARE catalogue releases and public roadmap meetings. Feedback is welcome via GitHub Discussions.

## Contributing

Chronos welcomes contributions from the community:

- Read the [CONTRIBUTING guidelines](CONTRIBUTING.md) for coding standards, review SLAs, and branching model.
- Sign the appropriate [Individual CLA](CLA/individual.md) or [Entity CLA](CLA/entity.md) before submitting pull requests.
- External contributors are encouraged to open issues describing proposed changes prior to implementation.

Pull requests trigger the full CI pipeline, including functional integration tests against mocked FIWARE services.

## Releases

- Semantic versioning (e.g., `1.2.0`) is enforced via Git tags.
- Each release is accompanied by notes in the [GitHub Releases](https://github.com/tcc-chronos/fiware-chronos/releases) section.
- FIWARE catalogue releases are mirrored using `FIWARE_<major>.<minor>` tags and multi-arch Docker images (`latest`, `<semver>`, `FIWARE_<major>.<minor>`).
- Release automation notifies the FIWARE infrastructure via `.github/fiware/image-clone.sh`.

Refer to the Release Management section in the Installation & Administration Guide for details.

## License

FIWARE Chronos is released under the [MIT License](LICENSE). When modifying the code, remember:

> Please note that software derived as a result of modifying the source code of the software in order to fix a bug or incorporate enhancements **IS** considered a derivative work of the product. Software that merely uses or aggregates (i.e. links to) an otherwise unmodified version of existing software **IS NOT** considered a derivative work.

© Chronos contributors, 2025. See `NOTICE` for third-party acknowledgements.
