<!--
  Contributing guidelines aligned with FIWARE Generic Enabler requirements.
-->

# Contributing to FIWARE Chronos

Thank you for your interest in improving Chronos. This guide explains how to propose changes, how intellectual property is handled, and the quality gates that protect end users. By contributing, you agree to follow the practices defined here.

## Governance & Community

- Chronos follows the [FIWARE Code of Conduct](https://www.fiware.org/code-of-conduct/).
- GitHub Issues is the primary tracker for bugs, feature requests, and questions.
- Issue triage occurs within **5 working days**. Please label issues with `bug`, `enhancement`, `question`, or `documentation`.
- Pull requests are reviewed within **10 working days**. Reviewers provide actionable feedback or merge the change. Emergency fixes are prioritised.

## Intellectual Property & Contributor License Agreement

Chronos is licensed under the MIT License. To protect contributors and adopters:

- All contributors **must** sign a Contributor License Agreement (CLA) before code can be merged.
  - [Individual CLA](CLA/individual.md)
  - [Entity CLA](CLA/entity.md)
- Submit the signed CLA via email to `chronos-cla@fiware.org`. We will acknowledge receipt within 3 working days.
- CLAs are adapted from the Harmony Agreements and comply with FIWARE clauses 2.1–2.3.
- The Chronos maintainers (Chronos Working Group) hold collective IPR on the project. If the maintainers cease support, IPR will be transferred to the FIWARE Foundation as described in the CLA.

By submitting a pull request you certify that you either own the code or have permission to contribute it under the CLA terms.

## Contribution Workflow

1. **Check existing issues** – avoid duplicates and comment if you plan to take an issue.
2. **Discuss major features** – open a design issue before coding or propose changes on the [roadmap](ROADMAP.md).
3. **Fork & branch** – create a feature branch from `main` (`feature/short-description`).
4. **Write tests & docs** – new functionality must include unit/integration coverage and documentation updates.
5. **Run quality checks:**
   ```bash
   make format
   make lint
   make test
   mkdocs build --strict
   ```
6. **Submit a pull request** – reference related issues (`Fixes #123`) and describe testing performed. Attach the CLA confirmation if not yet on record.
7. **Respond to review** – address comments promptly. Squash or amend commits as requested.

Pull requests that break CI will not be merged. All CI jobs must be green.

## Coding Standards

- Python code targets **Python 3.12** with type hints and `mypy --strict` compliance.
- Formatting is enforced via `black` and `isort`.
- Linting uses `flake8` with project configuration in `.flake8`.
- Tests use `pytest` with coverage ≥ 90% (configured in `pyproject.toml`).
- Public APIs must expose OpenAPI schemas. Update FastAPI tags and examples accordingly.

## Security & Responsible Disclosure

- Never include secrets in commits. Use environment variables or secret mounts.
- Report security vulnerabilities confidentially to `security@fiware.org` with the subject `FIWARE Chronos Security`.
- Confirmed vulnerabilities are triaged within **48 hours** with fixes released on an expedited schedule. Known issues are documented in the Installation & Administration Guide.

## Documentation Contributions

- User documentation resides under `docs/` and is published to Read the Docs.
- Update the relevant guide (User, Administration, API Reference) when behaviour changes.
- Use Markdown headings compliant with the FIWARE CSS guidelines. Avoid raw HTML anchors.
- Run `mkdocs build --strict` locally to detect broken links or formatting issues.

## Release Management

- Chronos follows semantic versioning: `MAJOR.MINOR.PATCH`.
- Each release candidate must pass full regression testing, including Docker image build and functional integration tests.
- Release notes are published under GitHub Releases and linked from the README.
- FIWARE catalogue tags (`FIWARE_<major>.<minor>`) are created after QA Lab validation.

## Questions?

- File a GitHub Issue tagged `question`.
- Reach the maintainers via the FIWARE Help Desk (`helpdesk.fiware.org`) and mention "Chronos".
- Participate in FIWARE roadmap meetings announced ahead of each FIWARE Summit.

We appreciate your contributions and commitment to the FIWARE ecosystem!
