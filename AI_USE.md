# AI use and citations

## AI-use disclosure

**Tool:** Claude Code (Anthropic), an AI coding assistant, used across many sessions from
2026-10-21 to 2026-11-15. Early milestones used Claude Sonnet 4.6; later sessions used other Claude
models, including Claude Opus 5.5. Some sessions ran in Anthropic's cloud sandbox, the rest locally.

**Kinds of assistance**

| Area | What the assistant did |
|---|---|
| Planning | Brainstorming the method, evaluation protocol and milestones; recording decisions in `docs/DECISIONS.md` |
| Code | Most of the code in `src/`, `scripts/`, `app/` and `tests/`: synthetic generator, calibration, HSV baseline, temporal filter and state machine, SQLite store, dashboard, SegFormer scripts, and the real-footage pipeline (registration, plate rectification, row features, gradient-boosting reader, decoding, quality gate, experiments, report) |
| Data | Searching for public data sources (`docs/DATA_SOURCES.md`), writing the USGS download and label scripts, drawing initial plate geometry |
| Experiments | Running the scripts that produce `results/` and logging runs in `docs/EXPERIMENT_LOG.md` |
| Debugging and review | Diagnosing failures, writing regression tests, reviewing the project |
| Writing | Documentation, and drafting `paper/main.tex` |
| Website | The project website in `site/` (Jekyll, Just the Docs theme), its asset-build script, the in-browser demo and its deployment workflow (2026-11-15) |

**What the student did and is responsible for:** choosing the project and its direction, deciding
on the changes of plan (for example the switch from synthetic data to real USGS footage), checking the
pipeline through the unit tests and by inspecting rectified strips, failure cases and split leakage,
and taking responsibility for every result and claim.

**Safeguards on the numbers.** No metric in the paper or the results files was typed by hand, by the
student or by the assistant. Every number is written by a script into `results/` and copied into the
paper as a LaTeX macro by `scripts/make_paper_assets.py`. Splits are committed and the locked test is
logged (`results/real/final_test_log.jsonl`), so the reported numbers can be regenerated and checked.

## Data used in the reported results

| Item | Source | Licence |
|---|---|---|
| Gauge-camera frames | USGS National Imagery Management System (NIMS), public S3 bucket `usgs-nims-images` | Public domain (US Government work) |
| Gauge height (labels) | USGS National Water Information System (NWIS) | Public domain (US Government work) |
| Gauge-height tables for in_overland and hancock | Copied from the committed result tables of [`tanayvasishtha/waterline`](https://github.com/tanayvasishtha/waterline) (Apache-2.0, commit a77a74c) and [`Waqar-msx9t/stagesight`](https://github.com/Waqar-msx9t/stagesight) (MIT, commit e9b10429). The values are USGS NWIS gauge readings; no code from these repositories is used | Apache-2.0 / MIT (the underlying values are public domain) |
| Initial plate geometry for in_overland | Started from the plate quadrilateral in waterline's `sites/in_overland.json`, then corrected by eye | Apache-2.0 |

Details and attribution text: `data/usgs/README.md`, `docs/DATA_SOURCES.md`.

## Software

| Library | Use | Licence |
|---|---|---|
| NumPy, SciPy | Arrays, numerics | BSD-3-Clause |
| scikit-learn (`HistGradientBoostingClassifier`) | Row classifier | BSD-3-Clause |
| OpenCV (SIFT, homography, image I/O) | Registration and rectification | Apache-2.0 |
| pandas, joblib | Tables, model serialisation | BSD-3-Clause |
| Matplotlib | Figures | Matplotlib licence (PSF-based) |
| Streamlit, Altair, streamlit-image-coordinates | Dashboard | Apache-2.0, BSD-3-Clause, MIT |
| Pillow, PyYAML | Images, configs | MIT-CMU, MIT |
| pytest, ruff | Tests, lint | MIT |
| NeurIPS 2026 LaTeX style (`paper/neurips_2026.sty`) | Paper layout, supplied unmodified by the course starter kit | NeurIPS author kit |

## Methods and papers cited

- D. G. Lowe, "Distinctive image features from scale-invariant keypoints" (SIFT), IJCV 60(2), 2004.
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python", JMLR 12, 2011.
- R. Vandaele et al., river-level monitoring from camera images, HESS 25, 2021.
- Y. Liang et al., V-FloodNet: urban flood video segmentation, Environmental Modelling & Software, 2023.

## Referenced but not used in any result

- SegFormer (Hugging Face `transformers`, Apache-2.0) with `nvidia/mit-b0` / `nvidia/mit-b1` weights:
  the training scripts exist (`scripts/train_seg.py`), but no weights were downloaded and no
  SegFormer result is reported. Check the model card licence before using the weights.
- ADE20K, COCO-Stuff and FloodNet: adapters exist in `src/floodcam/seg/`, but none of these datasets
  was downloaded. Confirm each licence on its official page before use.
- [`xmlyqing00/V-FloodNet`](https://github.com/xmlyqing00/V-FloodNet) and
  [`cvjena/eu-flood-dataset`](https://github.com/cvjena/eu-flood-dataset): cloned for reading only;
  no code or data from them is used.
