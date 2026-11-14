# FloodCam

Reading water depth from fixed-camera footage of a staff plate, evaluated on real USGS gauge cameras
against the official gauge height.

ES 678 (Machine Learning) course project, Kalp Shah, IIT Gandhinagar.
Paper: [`paper/main.pdf`](paper/main.pdf) (4 pages).

## Contents

| Path | What it holds |
|---|---|
| `src/floodcam/real/` | The method behind the reported results: SIFT registration and plate rectification (`plate.py`), row features, gradient-boosting reader and change-point decode (`reader.py`), cross-validation and baselines (`experiment.py`), metrics (`metrics.py`), inference API (`infer.py`) |
| `scripts/` | One script per step; the `real_*.py` scripts produce every reported number |
| `results/real/` | All reported outputs (metrics JSON, predictions CSV, figures, failure audit). Overview: [`results/real/SUMMARY.md`](results/real/SUMMARY.md) |
| `data/` | Committed: split files, gauge labels, a 68-frame sample. Frames are downloaded by script (see [Data access](#data-access)) |
| `models/` | Trained reader for in_overland (`reader.joblib`, 0.2 MB) and the registration reference views for each camera |
| `configs/sites/` | Plate geometry per camera |
| `app/` | Streamlit dashboard: replay of real USGS footage, upload-a-video mode, recorded runs |
| `paper/` | LaTeX source; every number is a macro generated from `results/` |
| `docs/` | `DECISIONS.md` (every non-obvious choice and why), `EXPERIMENT_LOG.md`, `DATA_SOURCES.md`, `FUTURE_GUIDE.md` (full technical guide) |
| `tests/` | Unit and integration tests |
| `src/floodcam/{synth,calib,depth,temporal,store,seg}` | The earlier synthetic-data pipeline and dashboard back end. Synthetic data is used only in tests and demos, never in reported results |

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The pinned versions in `requirements.txt` were checked on 2026-11-14 in a fresh environment
(Windows 11, Python 3.12). The trained model needs scikit-learn 1.9.1 and NumPy 2.x; older versions
cannot load it. PyTorch is **not** needed: it is only used by the optional SegFormer track, which
produced no reported result.

## Quick check (no download, a few minutes)

Everything here runs on committed files only.

```bash
python -m pytest -q tests/test_real.py     # real-footage pipeline: splits, model, demo frames
python -m pytest -q                        # full suite (~6 min; SegFormer tests skip without torch)
streamlit run app/dashboard.py             # choose "Real USGS footage" to replay a held-out event
```

Reading a frame with the trained model:

```python
import cv2
from floodcam.real.infer import PlateModel

model = PlateModel.load("models/in_overland/reader.joblib")
frame = cv2.imread("data/usgs/demo/frames/IN_Overland_Flow_East_of_Bull_Rapids_Road_near_Harlan_STAFF___2026-09-17T00-05-02Z.jpg")
r = model.read(frame)
print(r.status, r.depth_cm)        # status is "ok", "rejected_quality" or "unregistered"
```

The production model was trained on all in_overland frames, including the locked test months.
Use it for demos only; the held-out numbers are in `results/real/`.

## Data access

All data is public domain (US Government work) and needs no account, key or credentials.

**Committed in this repo (permitted sample):**

| Path | Content |
|---|---|
| `data/usgs/demo/` | 68 frames (640 px) of one held-out in_overland event (Sep 2026), with gauge height and the cross-validated estimate per frame (`demo.csv`) |
| `data/usgs/truth/*.csv` | Gauge labels for every frame used: `file,time,gage_ft` (about 9,850 rows over 6 cameras; `ny_owasco` was downloaded but not used) |
| `data/splits/usgs_*.json` | Month-block splits, including the locked test blocks (file IDs only) |
| `data/usgs/cameras.csv`, `camera_survey.csv` | List of the 1,353 USGS cameras and which staff-gauge cameras report gauge height |

**Full frames (downloaded by script, gitignored):**

```bash
python scripts/build_usgs_dataset.py --sites in_overland hancock in_north oh_wetland wi_mt1
```

This reads the file names in `data/usgs/truth/*.csv`, downloads each frame from the public S3 bucket
`https://usgs-nims-images.s3.amazonaws.com/overlay/<camera>/<file>`, resizes it to 1280 px and writes
`data/usgs/frames/` plus `data/usgs/manifest.jsonl`. About 8,450 frames, roughly 2 GB. Camera
folders are listed in `CAMERAS` in the script.

**Labels:** the in_overland and hancock gauge tables come from the result tables of two open-source
projects that queried USGS NWIS (sources and commits in `data/usgs/README.md`). The other cameras'
labels were fetched from the USGS APIs with `scripts/fetch_usgs_labels.py`. Gauge readings can also be
checked directly at `https://waterservices.usgs.gov/nwis/iv/?sites=<site_no>&parameterCd=00065`
(parameter 00065 = gage height in feet). Site numbers: in_overland 411229084541101,
hancock 01421500.

More detail: `data/usgs/README.md`, `docs/DATA_SOURCES.md`.

## Reproducing the results

After downloading the frames:

```bash
make real-strips      # SIFT registration + plate strips, 5 cameras and 6 downsampling factors (~1-2 h on 4 cores)
make real-eval        # CV per camera, resolution sweep, transfer to hancock, leave-one-camera-out, report
make paper            # regenerate paper/numbers.tex, tables, figures, then build main.pdf (needs pdflatex)
```

Without `make`, run the commands listed under those targets in the `Makefile`.

| Paper result | Script | Output |
|---|---|---|
| Main table: leave-one-month-out CV, baselines | `scripts/real_cv.py` | `results/real/real-cv-in_overland-f1/` |
| Locked test months | `scripts/real_final_test.py` | `results/real/real-final-in_overland-f1-force/`, `final_test_log.jsonl` |
| Per-camera table, in-site | `scripts/real_cv.py --site <camera>` | `results/real/real-cv-<camera>-f1/` |
| Per-camera table, camera held out | `scripts/real_multicam.py` | `results/real/loco/` |
| River camera (hancock) | `scripts/real_transfer.py` | `results/real/transfer-in_overland-to-hancock/` |
| Resolution sweep and error model | `scripts/real_resolution.py` | `results/real/resolution-in_overland/`, `real-res-*` |
| Figures, tables, failure audit | `scripts/real_report.py`, `scripts/make_paper_assets.py` | `results/real/figures/`, `tables/`, `failures/`, `paper/` |

Each run writes `meta.json` with the git commit, seed, manifest hash and split hash.

**Rules that are built into the code**

- Splits are by calendar month, never by frame. They are committed; do not regenerate them.
- The locked test months (2025-04, 2026-04) were fixed before training. `real_final_test.py` refuses
  a second run without `--force` and logs every run. The locked months were evaluated twice; the
  reason is disclosed in the paper and in `docs/DECISIONS.md`.
- Everything learned (dry template, classifier, calibration, baseline threshold, gate thresholds) is
  fitted on training months only.
- Seeds are fixed (`seed=0`).

The committed results were produced on Python 3.13 in a cloud sandbox. Reruns with the pinned
versions should give the same protocol and very close numbers. Bit-identical output across OpenCV
and scikit-learn versions has not been checked.

## Claim limits

The effective sample size is the number of months per camera, not the number of frames. Labels are a
gauge beside the plate, not tape measurements. Depth below the plate bottom (about 4 cm on
in_overland) cannot be read. The model does not transfer reliably to a new camera without that
camera's own labelled data (`results/real/loco/`). The Flood/Recovering state machine in the dashboard
is a demonstration, not an evaluated claim. The upload-a-video mode has been validated on synthetic
video only.

## AI use, citations and privacy

- AI-use disclosure, external data and code, licences and citations: [`AI_USE.md`](AI_USE.md).
- The repository contains no credentials, API keys or `.env` files. All USGS endpoints used are public
  and need no key.
- No private data: all images are public-domain USGS frames of gauge sites. No personal recordings
  or data about people are included.
