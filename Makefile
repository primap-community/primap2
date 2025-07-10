.PHONY: clean clean-test clean-pyc clean-build docs help virtual-environment install-pre-commit stubs update-venv README.md check-python-version
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
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: venv ## check style with pre-commit hooks
	venv/bin/pre-commit run --all-files

test: venv ## run tests quickly with the default Python
	venv/bin/pytest --xdoc -rx

test-all: ## run tests on every Python version with tox
	venv/bin/tox -p

coverage: venv ## check code coverage quickly with the default Python
	venv/bin/coverage run --source primap2 -m pytest
	venv/bin/coverage report -m
	venv/bin/coverage html
	ls htmlcov/index.html

clean-docs: venv ## Remove generated parts of documentation, then build docs
	. venv/bin/activate ; $(MAKE) -C docs clean
	. venv/bin/activate ; $(MAKE) -C docs html

docs: venv ## generate Sphinx HTML documentation, including API docs
	. venv/bin/activate ; $(MAKE) -C docs html

release: venv dist ## package and upload a release
	venv/bin/twine upload --repository primap dist/*

dist: clean venv ## builds source and wheel package
	# because we update the citation info after releasing on github and zenodo but
	# before building for pypi, we need to force the correct version.
	SETUPTOOLS_SCM_PRETEND_VERSION=0.12.3 venv/bin/python -m build

install: clean ## install the package to the active Python's site-packages
	python setup.py install

virtual-environment: venv ## setup a virtual environment for development

venv: requirements_dev.txt setup.cfg
	[ -d venv ] || python3 .check_python_version.py
	[ -d venv ] || python3 -m venv venv
	venv/bin/python -m pip install --upgrade wheel uv
	. venv/bin/activate ; venv/bin/uv pip install --upgrade -e .[dev]
	touch venv

update-venv: ## update all packages in the development environment
	[ -d venv ] || python3 -m venv venv
	venv/bin/python .check_python_version.py
	venv/bin/python -m pip install --upgrade wheel uv
	. venv/bin/activate ; venv/bin/uv pip  install --upgrade --resolution highest -e .[dev]
	touch venv

install-pre-commit: update-venv ## install the pre-commit hooks
	venv/bin/pre-commit install

stubs: venv ## generate directory with xarray stubs with inserted primap2 stubs
	rm -rf stubs
	mkdir -p stubs
	venv/bin/stubgen -p xarray -o stubs
	(cd stubs; patch -s -p0 < ../primap-stubs.patch)

README.md: ## Update the citation information from zenodo
	venv/bin/python update_citation_info.py
