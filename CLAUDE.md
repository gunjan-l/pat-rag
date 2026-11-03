# PAT AI Platform - Open-Source RAG Core

## Environment & Tooling Rules (STRICT)

- **NEVER install Python packages, `uv`, or runtime tools on the Windows host.**
- **ALWAYS use Podman containers**: Pull container images from registries (e.g. `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, `docker.io/qdrant/qdrant:latest`, `docker.io/langfuse/langfuse:latest`).
- All package resolution, test runs, linting, and application execution run isolated inside Podman containers via volume mounts (`-v ${PWD}:/app:z`).
- Windows host remains completely clean.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context domain documentation (`GLOSSARY.md`, `docs/adr/`). See `docs/agents/domain.md`.

