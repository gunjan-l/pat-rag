param (
    [Parameter(Position=0)]
    [ValidateSet("up", "down", "shell", "run", "debug", "test", "lint", "stop")]
    [string]$Command = "run"
)

$ContainerImage = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
$AppDir = "${PWD}"
$ContainerName = "pat-rag-api-dev"

switch ($Command) {
    "up" {
        Write-Host "Starting all services with podman-compose..." -ForegroundColor Cyan
        podman-compose up -d
    }
    "down" {
        Write-Host "Stopping all services with podman-compose..." -ForegroundColor Cyan
        podman-compose down
    }
    "stop" {
        Write-Host "Stopping dev container..." -ForegroundColor Cyan
        podman stop $ContainerName -t 2 2>$null
    }
    "shell" {
        Write-Host "Opening interactive shell inside dev container..." -ForegroundColor Cyan
        podman run -it --rm -v "${AppDir}:/app:z" -w /app -p 8000:8000 --env-file .env $ContainerImage bash
    }
    "run" {
        Write-Host "Starting FastAPI dev server with auto-reload on port 8000..." -ForegroundColor Cyan
        podman stop $ContainerName -t 2 2>$null
        podman run -it --rm --name $ContainerName -v "${AppDir}:/app:z" -w /app -p 8000:8000 --env-file .env $ContainerImage uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "debug" {
        Write-Host "Starting FastAPI with debugpy on port 5678 & 8000..." -ForegroundColor Cyan
        podman stop $ContainerName -t 2 2>$null
        podman run -it --rm --name $ContainerName -v "${AppDir}:/app:z" -w /app -p 8000:8000 -p 5678:5678 --env-file .env $ContainerImage uv run python -m debugpy --listen 0.0.0.0:5678 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "test" {
        Write-Host "Running pytest inside container..." -ForegroundColor Cyan
        podman run --rm -v "${AppDir}:/app:z" -w /app $ContainerImage uv run pytest -v
    }
    "lint" {
        Write-Host "Running ruff check inside container..." -ForegroundColor Cyan
        podman run --rm -v "${AppDir}:/app:z" -w /app $ContainerImage uv run ruff check .
    }
}
