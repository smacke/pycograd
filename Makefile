# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy_only deploy black blackcheck imports lint typecheck check_no_typing check test tests coverage xmlcov check_ci devdeps

# Prefer the project virtualenv's tools (mypy, ruff, isort, pytest, ...) when a
# .venv is present, so `make` targets work without activating it. CI installs
# into the host Python and has no .venv, so VBIN is empty there and the bare tool
# names resolve via PATH as before. (A path prefix rather than an exported PATH
# because GNU Make 3.81 — macOS's default — execs single-word recipes directly
# using its startup PATH and ignores `export PATH`.)
VBIN := $(if $(wildcard .venv/bin),.venv/bin/,)

clean:
	rm -rf __pycache__ build/ dist/ *.egg-info/ .coverage htmlcov

build: clean
	$(VBIN)python -m build

bump:
	./scripts/bump-version.py

deploy_only:
	./scripts/deploy.sh

deploy: build deploy_only

black:
	$(VBIN)isort .
	BLACK=$(VBIN)black ./scripts/blacken.sh

blackcheck:
	$(VBIN)isort . --check-only
	BLACK=$(VBIN)black ./scripts/blacken.sh --check

imports:
	$(VBIN)pycln .
	$(VBIN)isort .

lint:
	$(VBIN)ruff check

typecheck:
	$(VBIN)mypy pycograd

check_no_typing:
	rm -f .coverage
	rm -rf htmlcov
	$(VBIN)pytest --cov-config=pyproject.toml --cov=pycograd

check: blackcheck lint typecheck check_no_typing

test: check
tests: check

coverage: check_no_typing
	$(VBIN)coverage html

xmlcov: check_no_typing
	$(VBIN)coverage xml

check_ci: typecheck xmlcov

devdeps:
	$(VBIN)pip install -e .[dev]
