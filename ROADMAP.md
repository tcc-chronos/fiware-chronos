<!--
  FIWARE Generic Enabler roadmap following GE_roadmap_template.md
-->

# FIWARE Chronos Roadmap

## Vision

Chronos enables FIWARE solutions to forecast IoT sensor behaviour with explainable, maintainable machine learning pipelines. The roadmap aligns with FIWARE catalogue releases and NGSI-LD adoption efforts.

## Strategic Themes

- **Operational Excellence** – ensure reliability, observability, and security for production deployments.
- **Forecasting Quality** – expand modelling techniques and evaluation tooling.
- **Ecosystem Integration** – embrace NGSI-LD features, Smart Data Models, and interoperability with other GEs.

## Release Plan

| Timeframe | Release | Focus Areas | Notes |
|-----------|---------|-------------|-------|
| 2025 Q1   | 1.0.0 (FIWARE_1.0) | Initial FIWARE GE publication, Orion v2 integration, Docker distributions, documentation on Read the Docs. | Target QA label **A**. |
| 2025 Q2   | 1.1.0 | Automated hyperparameter tuning, improved training dashboards, OpenTelemetry traces. | Requires upgrades to TensorFlow serving image. |
| 2025 Q3   | 1.2.0 (FIWARE_1.1) | NGSI-LD entity support, dual Orion/Scorpio compatibility, Smart Data Models templates. | Presented at FIWARE Global Summit roadmap session. |
| 2025 Q4   | 1.3.0 | Multi-tenant scheduling, anomaly detection add-ons, predictive maintenance tutorials. | Milestones reviewed with FIWARE QA Lab. |

Roadmap updates are discussed at each FIWARE Summit F2F session. Feedback from the community is incorporated through GitHub Discussions and the FIWARE Help Desk.

## Backlog Candidates

- Federated learning for distributed edge devices.
- GPU-enabled training pipelines.
- Plug-in architecture for custom model evaluators and explainability dashboards.
- Automated dataset drift detection with Orion subscriptions.

## Dependencies & Risks

- Synchronisation with Orion Context Broker and Scorpio releases.
- Availability of reference datasets for benchmarking.
- Resource coordination for FIWARE QA Lab validation windows.

## Contact

Roadmap questions can be addressed to the Chronos Working Group via:

- GitHub Discussions (`Q&A` category).
- FIWARE Help Desk (subject: "Chronos Roadmap").
- FIWARE Processing & Analysis Chapter meetings.
