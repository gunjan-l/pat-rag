#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-run}"
CONTAINER_IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
APP_DIR="$(pwd)"
CONTAINER_NAME="pat-rag-api-dev"

case "$COMMAND" in
    up)
        echo "Starting all services with podman-compose..."
        podman-compose up -d
        ;;
    down)
        echo "Stopping all services with podman-compose..."
        podman-compose down
        ;;
    stop)
        echo "Stopping dev container..."
        podman stop -t 2 "$CONTAINER_NAME" 2>/dev/null || true
        ;;
    shell)
        echo "Opening interactive shell inside dev container..."
        podman run -it --rm \
            -v "${APP_DIR}:/app:z" \
            -w /app \
            -p 8000:8000 \
            --env-file .env \
            "$CONTAINER_IMAGE" bash
        ;;
    run)
        echo "Starting FastAPI dev server with auto-reload on port 8000..."
        podman stop -t 2 "$CONTAINER_NAME" 2>/dev/null || true
        podman run -it --rm \
            --name "$CONTAINER_NAME" \
            -v "${APP_DIR}:/app:z" \
            -w /app \
            -p 8000:8000 \
            --env-file .env \
            "$CONTAINER_IMAGE" \
            uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    debug)
        echo "Starting FastAPI with debugpy on port 5678 & 8000..."
        podman stop -t 2 "$CONTAINER_NAME" 2>/dev/null || true
        podman run -it --rm \
            --name "$CONTAINER_NAME" \
            -v "${APP_DIR}:/app:z" \
            -w /app \
            -p 8000:8000 \
            -p 5678:5678 \
            --env-file .env \
            "$CONTAINER_IMAGE" \
            uv run python -m debugpy --listen 0.0.0.0:5678 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    test)
        echo "Running pytest inside container..."
        podman run --rm \
            -v "${APP_DIR}:/app:z" \
            -w /app \
            "$CONTAINER_IMAGE" \
            uv run pytest -v
        ;;
    lint)
        echo "Running ruff check inside container..."
        podman run --rm \
            -v "${APP_DIR}:/app:z" \
            -w /app \
            "$CONTAINER_IMAGE" \
            uv run ruff check .
        ;;
    *)
        echo "Usage: ./scripts/dev.sh [run|shell|debug|up|down|stop|test|lint]"
        exit 1
        ;;
esac
