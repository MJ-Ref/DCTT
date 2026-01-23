.PHONY: install install-dev install-all test test-fast test-cov lint format typecheck clean help

# Default Python interpreter
PYTHON ?= python3

# Package name
PACKAGE = dctt

help:
	@echo "DCTT Development Commands"
	@echo "========================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package in production mode"
	@echo "  make install-dev   Install package with dev dependencies"
	@echo "  make install-all   Install package with all optional dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests excluding slow markers"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-unit     Run only unit tests"
	@echo "  make test-prop     Run only property-based tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with ruff"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make check         Run all checks (lint + typecheck)"
	@echo ""
	@echo "Development:"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make pre-commit    Install pre-commit hooks"
	@echo ""
	@echo "Experiments:"
	@echo "  make census        Run diagnostic census experiment"
	@echo "  make repair        Run causal repair experiment"

# Installation targets
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[all]"

# Testing targets
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src/$(PACKAGE) --cov-report=html --cov-report=term-missing

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-prop:
	$(PYTHON) -m pytest tests/property/ -v -m property

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v -m integration

# Code quality targets
lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck:
	$(PYTHON) -m mypy src/$(PACKAGE)

check: lint typecheck

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Experiment targets
census:
	$(PYTHON) -m experiments.run_census

repair:
	$(PYTHON) -m experiments.run_causal_repair

ablation:
	$(PYTHON) -m experiments.run_ablation_suite

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf outputs/coverage/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker (optional)
docker-build:
	docker build -t dctt:latest .

docker-run:
	docker run -it --rm -v $(PWD):/workspace dctt:latest
