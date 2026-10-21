# FloodCam Makefile
# On Windows: run from Git Bash or WSL, or adapt targets for PowerShell.

PYTHON := python
PYTEST := python -m pytest

.PHONY: setup test test-fast synth-demo train-seg eval dashboard lint format clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -x

synth-demo:
	$(PYTHON) -m floodcam.synth.demo

train-seg:
	$(PYTHON) scripts/train_seg.py --config configs/default.yaml

eval:
	$(PYTHON) scripts/eval.py --config configs/default.yaml

dashboard:
	streamlit run app/dashboard.py

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
