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

.PHONY: sgr/ollama
sgr/ollama:
	inspect eval examples/structured.py@rgb_color --model ollama/qwen3:4b-fp16

.PHONY: sgr/openrouter
sgr/openrouter:
	inspect eval examples/structured.py@rgb_color --model openrouter/qwen/qwen3-235b-a22b:free

# https://abdullin.com/schema-guided-reasoning/
.PHONY: sgr
experiment: sgr/ollama sgr/openrouter
