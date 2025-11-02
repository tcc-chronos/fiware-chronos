# Test Report

This template summarizes the results of a test cycle. Replace placeholders with actual execution results.

## Summary

- Date: <YYYY-MM-DD>
- Commit: <git SHA>
- Environment: <local/CI>
- Scope: <features/modules>

## Results

- Total tests: <N>
- Passed: <N>
- Failed: <N>
- Skipped: <N>
- Coverage (lines): <XX.xx%> (threshold: 90%)

## Breakdown by Test Type

- Unit: <counts>, key focus: <areas>
- Integration: <counts>, components: <repos/gateways>
- Functional: <counts>, scenarios: <flows>
- E2E: <counts>, endpoints: `/health`, `/info`, …

## Notable Scenarios

- Model lifecycle: create → train → predict → toggle recurring
- Orion publish failure handling (retry/backoff)
- STH‑Comet pagination across large datasets

## Defects

- ID: <#>, Severity: <low/medium/high>, Status: <open/fixed>, Summary: <...>

## Attachments

- coverage.xml, htmlcov/ (attach or link)
- CI job URL: <link>
