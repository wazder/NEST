.PHONY: help install install-dev test test-unit test-integration coverage lint format clean docs

# Default target
help:
	@echo "NEST - Neural EEG Sequence Transducer"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install project dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-fast        Run fast tests (exclude slow tests)"
	@echo "  coverage         Run tests with coverage report"
	@echo "  lint             Run all linters"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  docs             Build documentation"
	@echo "  pre-commit       Install pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

test-fast:
	pytest -m "not slow" -v

test-parallel:
	pytest -n auto

coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	@echo "Running flake8..."
	flake8 src tests
	@echo "Running pylint..."
	pylint src --exit-zero
	@echo "Checking code complexity..."
	radon cc src -a -nb
	@echo "Linting complete!"

format:
	@echo "Formatting with black..."
	black src tests
	@echo "Sorting imports with isort..."
	isort src tests
	@echo "Formatting complete!"

format-check:
	black --check src tests
	isort --check-only src tests

type-check:
	mypy src --ignore-missing-imports

security:
	@echo "Running bandit security scan..."
	bandit -r src -ll
	@echo "Running safety check..."
	safety check

# Pre-commit
pre-commit:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run:
	pre-commit run --all-files

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/index.html"

docs-serve:
	@echo "Serving documentation on http://localhost:8000"
	cd docs/_build/html && python -m http.server

# Quick development commands
dev-setup: install-dev pre-commit
	@echo "Development environment setup complete!"

check-all: format lint type-check security test
	@echo "All checks passed!"

# Docker commands (if needed)
docker-build:
	docker build -t nest:latest .

docker-run:
	docker run -it --rm nest:latest

# Dataset download (optional)
download-zuco:
	python -m src.data.zuco_dataset --download --data-dir data/raw/zuco
