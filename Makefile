.PHONY: help run dev test lint format clean docker-up docker-down install

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Dependencies ──────────────────────────────────────────
install: ## Install all dependencies (including dev)
	pip install -e ".[dev]"

install-prod: ## Install production dependencies only
	pip install -e .

# ── Infrastructure ────────────────────────────────────────
docker-up: ## Start Chroma + Redis
	docker compose up -d

docker-down: ## Stop Chroma + Redis
	docker compose down

# ── Application ───────────────────────────────────────────
run: ## Run the API server (prod mode)
	uvicorn app.main:app --host $(HOST) --port $(PORT)

dev: ## Run with hot-reload (development)
	uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

# ── Code Quality ──────────────────────────────────────────
lint: ## Lint with ruff
	ruff check app/ tests/

format: ## Auto-format with ruff
	ruff check --fix app/ tests/
	ruff format app/ tests/

typecheck: ## Static type checking
	mypy app/

# ── Testing ───────────────────────────────────────────────
test: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=app --cov-report=term-missing

test-watch: ## Run tests on file change
	ptw tests/ app/ -- -v

# ── Database ──────────────────────────────────────────────
migrate: ## Run Alembic migrations
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new msg="add users table")
	alembic revision --autogenerate -m "$(msg)"

# ── Cleanup ───────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache dist build
