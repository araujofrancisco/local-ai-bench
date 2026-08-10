.PHONY: help build up down logs clean frontend-dev backend-dev test

help:
	@echo "OllamaBench Docker + Astro Deployment"
	@echo ""
	@echo "Available targets:"
	@echo "  make build         - Build Docker images"
	@echo "  make up            - Start all services (detached)"
	@echo "  make down          - Stop all services"
	@echo "  make logs          - View logs"
	@echo "  make clean         - Stop and remove volumes"
	@echo "  make frontend-dev  - Run Astro dev server locally"
	@echo "  make backend-dev   - Run FastAPI locally"
	@echo "  make test          - Run tests"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v

frontend-dev:
	cd web && npm install && npm run dev

backend-dev:
	uvicorn ollama_bench.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest
