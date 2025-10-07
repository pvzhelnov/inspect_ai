.PHONY: hooks
hooks:
	pre-commit install

.PHONY: conda
conda:
	conda env create -f environment.yml

.PHONY: install
install:
	uv pip install --python $$(which python) -r requirements.txt
	uv pip install --python $$(which python) -e .[dev]

.PHONY: ruff
ruff:
	ruff check --fix
	ruff format

.PHONY: mypy
mypy:
	mypy --exclude tests/test_package src tests

.PHONY: check
check: ruff mypy
	pylint src

.PHONY: test
test:
	pytest
