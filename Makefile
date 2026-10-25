# FloodCam Makefile
# On Windows: run from Git Bash or WSL, or adapt targets for PowerShell.

PYTHON := python
PYTEST := python -m pytest

.PHONY: upload-demo setup test test-fast synth-demo calib-headless baseline temporal-demo run-demo run-demo-multi train-seg train-seg-smoke eval-seg eval dashboard lint format clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -x

synth-demo:
	$(PYTHON) -m floodcam.synth.demo --out-dir data/synthetic --seed 42
	$(PYTHON) scripts/generate_synthetic_splits.py \
		--manifest data/synthetic/manifest.jsonl \
		--seed 42 --force

calib-headless:
	$(PYTHON) -m floodcam.calib.annotate \
		--points-json configs/example_calib_points.json \
		--camera-id example_cam \
		--output configs/example_cam.yaml \
		--mode linear

baseline:
	$(PYTHON) scripts/run_hsv_baseline.py --run-id hsv-001

temporal-demo:
	$(PYTHON) scripts/simulate_state_machine.py --run-id temporal-001

run-demo:
	$(PYTHON) scripts/run_demo.py --run-id run-001

run-demo-multi:
	$(PYTHON) scripts/run_demo_multi.py --run-id demo-multi-001

train-seg:
	$(PYTHON) scripts/train_seg.py --config configs/seg_pretrain.yaml

train-seg-smoke:
	$(PYTHON) scripts/train_seg.py --smoke --run-id seg-smoke-001 --overwrite

eval-seg:
	$(PYTHON) scripts/eval_seg.py --backend hsv --split val --with-depth

eval:
	$(PYTHON) scripts/eval.py --config configs/default.yaml

dashboard:
	$(PYTHON) scripts/run_demo_multi.py --run-id demo-multi-001
	streamlit run app/dashboard.py

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

upload-demo:
	streamlit run app/dashboard.py
