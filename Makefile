PYTHON ?= .venv/bin/python
UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help setup env db-up db-down db-init data data-weather weather load pipeline test test-unit lint format clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv (Python 3.12) and install the project + dev tools
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env - edit POSTGRES_PASSWORD")

db-up: env ## Start PostgreSQL (docker compose)
	docker compose up -d --wait postgres

db-down: ## Stop all compose services
	docker compose down

db-init: ## Create the raw schema and tables
	$(PYTHON) -m src.db.init_db

data: ## Generate the synthetic dataset (offline, synthetic weather)
	$(PYTHON) -m src.ingestion.generate_data

data-weather: ## Generate synthetic demand driven by real Open-Meteo weather
	$(PYTHON) -m src.ingestion.generate_data --weather open-meteo

weather: ## Download Nouakchott historical weather from Open-Meteo
	$(PYTHON) -m src.ingestion.weather_api

load: ## Validate and load the synthetic dataset into PostgreSQL
	$(PYTHON) -m src.ingestion.load_data --source synthetic

pipeline: db-up data load ## Start DB, generate data, load it

test: ## Run all tests (DB integration tests skip if PostgreSQL is down)
	$(PYTHON) -m pytest

test-unit: ## Run tests that do not need PostgreSQL
	$(PYTHON) -m pytest -m "not integration"

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format: ## Auto-format and fix lint issues
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

clean: ## Remove caches (keeps data and the database)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
