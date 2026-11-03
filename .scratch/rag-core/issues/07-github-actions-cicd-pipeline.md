# 07: GitHub Actions CI/CD Pipeline

**What to build:** A GitHub Actions workflow (`.github/workflows/ci.yml`) that validates pull requests and pushes by running code formatting/linting, type checking, and the automated test suite with mocked external APIs.

**Blocked by:** 02: Document Extraction Service (Component 1), 03: Hybrid Vector Storage & Indexing Service (Component 2), 04: Agentic Retrieval, Confidence Fallback & Query Rewrite (Component 3)

**Status:** ready-for-agent

- [ ] GitHub Actions workflow file `.github/workflows/ci.yml` triggers on push to main and all pull requests.
- [ ] Workflow checks out code, sets up Python 3.12, and caches package dependencies.
- [ ] Linter (ruff) and code quality checks run and gate the pipeline on style or syntax errors.
- [ ] Automated test suite runs with pytest, exercising contract tests and mocked external API tests.
- [ ] Container build smoke test verifies the application image builds cleanly.
