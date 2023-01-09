#!/bin/bash

DIR := $(shell pwd)
export PYTHONPATH := $(DIR)/src

# Virtual environment commands
venv:
	python -m venv ./venv || true
	. ./activate_venv && python -m pip install -q pip wheel --upgrade;
	. ./activate_venv && python -m pip install -q -r requirements-dev.txt

update:
	. ./activate_venv && python -m pip install -q -r requirements-dev.txt

# Test commands
test:
	. ./activate_venv && pytest --cov-report term --cov=src

# Examples commands
jupyter-examples:
	. ./activate_venv && cd examples && jupyter notebook

.PHONY: venv