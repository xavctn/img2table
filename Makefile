#!/bin/bash

DIR := $(shell pwd)
export PYTHONPATH := $(DIR)/src

# Virtual environment commands
venv:
	uv sync --all-extras --dev

update:
	uv sync --upgrade --all-extras --dev

# Test commands
test:
	uv run pytest --cov-report term --cov=src

fast-test:
	uv run pytest --cov-report term --cov=src --ignore=tests/ocr --ignore=tests/document/base

lint:
	uv run ruff check

type-check:
	uv run ty check src tests

# Examples commands
jupyter:
	cd examples && uv run jupyter notebook

update-examples:
	for f in $(PWD)/examples/*.ipynb; do \
	  uv run jupyter nbconvert --to notebook --execute $$f --inplace; \
	done

# Build commands
build: venv
	uv build

.PHONY: venv