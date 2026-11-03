# 01: Project Scaffolding & Container Environment

**What to build:** A working, configured base application structure with dependencies, configuration loading from `.env`, FastAPI application entrypoint with healthcheck endpoint, and container orchestration via Podman Compose for Qdrant and Langfuse.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] Project dependency specifications (`pyproject.toml` and `requirements.txt`) define FastAPI, Uvicorn, LangChain, Qdrant client, FastEmbed, Docling, Langfuse, Ragas, and testing tools.
- [x] Containerfile / Podman dev environment configured using `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` so zero libraries are installed on Windows host.
- [x] Centralized configuration module using Pydantic Settings reads environment variables with sensible defaults for local development.
- [x] FastAPI application initializes with CORS and exposes a `GET /health` endpoint returning operational status.
- [x] `podman-compose.yml` provides reproducible local services for Qdrant vector database (port 6333) and Langfuse with PostgreSQL.
- [x] Automated healthcheck test verifies the application boots cleanly.
