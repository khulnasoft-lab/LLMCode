.PHONY: help install install-dev install-browser install-all test test-cov lint format check-codestyle clean clean-build clean-pyc clean-test

# Default target when running just 'make'
help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install           Install the package and required dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  install-browser   Install browser automation dependencies"
	@echo "  install-all       Install all dependencies (main, dev, browser)"
	@echo "  test              Run tests"
	@echo "  test-cov          Run tests with coverage report"
	@echo "  lint              Check code style with flake8"
	@echo "  format            Format code with black and isort"
	@echo "  check-codestyle   Check code style without making changes"
	@echo "  clean             Remove all build, test, coverage and Python artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-browser:
	pip install -e ".[browser]"

install-all: install install-dev install-browser

# Testing
test:
	pytest tests/

test-cov:
	pytest --cov=llmcode --cov-report=term-missing tests/

# Linting and formatting
lint:
	flake8 llmcode tests

format:
	isort llmcode tests
	black llmcode tests

check-codestyle:
	isort --check-only llmcode tests
	black --check llmcode tests

# Cleanup
clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/ dist/ .eggs/ 
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {}

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .pytest_cache
	rm -f .coverage

# Build and release
build:
	python -m build

release: clean build
	twine upload dist/*

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files
