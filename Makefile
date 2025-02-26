#!/bin/bash

VENV = ./activate_venv
DIR := $(shell pwd)
export PYTHONPATH := $(DIR)/src

# Virtual environment commands
venv:
	python -m venv ./venv || true
	. $(VENV); python -m pip install pip wheel --upgrade;
	. $(VENV); python -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

update:
	. $(VENV); python -m pip install -U -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Test commands
test:
	. $(VENV); pytest --cov-report term --cov=src

# Examples commands
jupyter-examples:
	. $(VENV); cd examples && jupyter notebook

update-examples:
	. $(VENV);
	for f in $(PWD)/examples/*.ipynb; do \
	  jupyter nbconvert --to notebook --execute $$f --inplace; \
	done

# Build commands
build: venv
	. $(VENV); python setup.py sdist bdist_wheel


.PHONY: venv