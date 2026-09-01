.PHONY: clean clean-test clean-pyc clean-build docs help virtual-environment install-pre-commit stubs update-venv README.md
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .venv-test-*/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: .venv ## check style with pre-commit hooks
	uv run pre-commit run --all-files

test: .venv ## run tests quickly with the default Python
	uv run pytest --xdoc -rx

test-all: ## run tests on every supported Python version and dependency resolution
	@for python in 3.11 3.12; do \
	  for resolution in highest lowest-direct; do \
	    echo "=== Python $$python, $$resolution dependency resolution ==="; \
	    uv venv --python $$python .venv-test-$$python-$$resolution || exit 1; \
	    VIRTUAL_ENV=.venv-test-$$python-$$resolution uv pip install --resolution $$resolution ".[test]" || exit 1; \
	    .venv-test-$$python-$$resolution/bin/pytest --xdoc -rx || exit 1; \
	  done; \
	done

coverage: .venv ## check code coverage quickly with the default Python
	uv run coverage run --source primap2 -m pytest
	uv run coverage report -m
	uv run coverage html
	ls htmlcov/index.html

clean-docs: .venv ## Remove generated parts of documentation, then build docs
	uv run $(MAKE) -C docs clean
	uv run $(MAKE) -C docs html

docs: .venv ## generate Sphinx HTML documentation, including API docs
	uv run $(MAKE) -C docs html

release: .venv dist ## package and upload a release
	uv run twine upload --repository primap dist/*

dist: clean .venv ## builds source and wheel package
	uv build

virtual-environment: .venv ## setup a virtual environment for development

.venv: pyproject.toml uv.lock ## create or update the development virtual environment
	uv sync
	@touch .venv

update-venv: ## update all packages in the development environment
	uv sync --upgrade
	@touch .venv

install-pre-commit: .venv ## install the pre-commit hooks
	uv run pre-commit install

stubs: .venv ## generate directory with xarray stubs with inserted primap2 stubs
	rm -rf stubs
	mkdir -p stubs
	uv run stubgen -p xarray -o stubs
	(cd stubs; patch -s -p0 < ../primap-stubs.patch)

README.md: ## Update the citation information from zenodo
	uv run python update_citation_info.py
