PY ?= python

.PHONY: demo test lint benchmark clean

demo:
	$(PY) -m survforge.cli demo --out benchmark.json

benchmark:
	$(PY) -m survforge.cli benchmark --out benchmark.json

test:
	$(PY) -m pytest -q -W ignore::UserWarning

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache survforge.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
