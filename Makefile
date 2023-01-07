#!/bin/bash

args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

venv:
	python -m venv ./venv || true
	. ./activate_venv && python -m pip install -q pip --upgrade;
	. ./activate_venv && python -m pip install -q -r requirements-dev.txt

update: venv
	. ./activate_venv && python -m pip install -q -r requirements-dev.txt

test:
	. ./activate_venv && pytest --cov-report term --cov=src

.PHONY: venv