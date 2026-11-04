FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dependency definition
COPY pyproject.toml requirements.txt /app/

# Sync dependencies without project root
RUN uv sync --no-install-project

# Copy source code
COPY src/ /app/src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
