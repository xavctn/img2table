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
	uv run pytest --cov --cov-config=pyproject.toml --cov-report=term

fast-test:
	uv run pytest --cov --cov-config=pyproject.toml --cov-report=term --ignore=tests/ocr --ignore=tests/document/rotation

lint:
	uv run ruff check

type-check:
	uv run ty check src tests

# Examples commands
jupyter: build-ext
	cd examples && uv run jupyter notebook

update-examples:
	for f in $(PWD)/examples/*.ipynb; do \
	  uv run jupyter nbconvert --to notebook --execute $$f --inplace; \
	done

# Build commands
build:
	uv build

build-ext:
	python setup.py build_ext --inplace

clean-ext:
	find src -type f \( -name "*.pyd" -o -name "*.so" \) -delete
	find src -type f -name "*.pyx" -exec sh -c 'rm -f "$${1%.pyx}.c"' _ {} \;
	rm -rf build

.PHONY: venv build build-ext clean-ext
