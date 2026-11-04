param (
    [Parameter(Position=0)]
    [ValidateSet("up", "down", "shell", "run", "test", "lint")]
    [string]$Command = "run"
)

$ContainerImage = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
$AppDir = "${PWD}"

switch ($Command) {
    "up" {
        Write-Host "Starting all services with podman-compose..." -ForegroundColor Cyan
        podman-compose up -d
    }
    "down" {
        Write-Host "Stopping all services with podman-compose..." -ForegroundColor Cyan
        podman-compose down
    }
    "shell" {
        Write-Host "Opening interactive shell inside dev container..." -ForegroundColor Cyan
        podman run -it --rm -v "${AppDir}:/app:z" -w /app -p 8000:8000 -p 5678:5678 --env-file .env $ContainerImage bash
    }
    "run" {
        Write-Host "Starting FastAPI dev server with auto-reload on port 8000..." -ForegroundColor Cyan
        podman run -it --rm -v "${AppDir}:/app:z" -w /app -p 8000:8000 -p 5678:5678 --env-file .env $ContainerImage uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
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
