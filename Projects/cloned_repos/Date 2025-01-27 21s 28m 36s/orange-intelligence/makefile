PROJECT ?= orange-intelligence
VERSION ?= $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
PYTHON_VERSION = $(shell grep 'python =' pyproject.toml | sed -n 's/^python = ["^]*\([0-9]*\.[0-9]*\)\(.*\)"/\1/p')

version: ## Show the current version
	echo "Current version: $(VERSION)"

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort |\
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL=help

.python-version: ## Installs the correct version of python if required
	$(if $(shell pyenv versions --bare | grep "^$(PYTHON_VERSION)"),, pyenv install $(PYTHON_VERSION))
	pyenv local $(shell pyenv versions --bare | grep "^$(PYTHON_VERSION)" | tail -n 1)

.venv: .python-version ## Activates and installs the poetry environment.
	poetry env use ~/.pyenv/versions/$(shell cat $<)/bin/python && \
	poetry install

lint: .venv ## Lint the project
	poetry run flake8

mypy: .venv ## Run mypy
	poetry run mypy

test: .venv ## Test the project
	poetry run pytest

fmt: .venv ## Run formatting tools for project
	poetry run black . $(if $(CI),--check ,) && poetry run isort . $(if $(CI),--check ,)

bandit: .venv ## Run bandit
	poetry run bandit -c pyproject.toml -r .

check: fmt lint mypy bandit test ; ## Run all checks

run:
	poetry run python3 app.py

clean: ## Clean the project
	rm -rf .venv
	rm -rf .python-version
	rm -rf dist/
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
