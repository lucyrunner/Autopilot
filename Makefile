.PHONY: install run test clean docker-build docker-run

install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/base.txt
	pip install -r requirements/prod.txt
	pip install pytest pytest-cov flake8 mypy black isort

run:
	python -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

worker:
	python -m src.inference.worker --worker-id 0

lint:
	flake8 src/ --max-line-length=100
	black --check src/
	isort --check-only src/

format:
	black src/
	isort src/

docker-build:
	docker build -f docker/Dockerfile.cpu -t autopilot-vision:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

# Type checking and linting
check: lint typecheck test
	@echo "✅ All checks passed!"

lint:
	@echo "🔍 Running ruff..."
	ruff check src/

lint-fix:
	@echo "🔧 Fixing with ruff..."
	ruff check --fix src/

typecheck:
	@echo "🔍 Running mypy..."
	mypy src/ --ignore-missing-imports

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-quick:
	@echo "🧪 Running tests (quick)..."
	pytest tests/ -q

test-coverage:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "📊 Coverage report: htmlcov/index.html"
