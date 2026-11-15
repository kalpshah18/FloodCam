---
title: Reproduce
layout: default
nav_order: 7
description: "Setup, data download, make targets, which script produces which result, and how this website is built."
---

# Reproduce
{: .no_toc }

1. TOC
{:toc}

Everything lives in the repository: [github.com/kalpshah18/FloodCam](https://github.com/kalpshah18/FloodCam). Each result comes from a script and a saved results file. Nothing reported is typed by hand.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The trained model needs scikit-learn 1.9.1 and NumPy 2.x; older versions cannot load it. PyTorch is not needed: it is only used by an optional SegFormer track that produced no reported result.

## Quick check (committed files only)

```bash
python -m pytest -q tests/test_real.py     # splits, model, demo frames
python -m pytest -q                        # full suite, about 6 minutes
streamlit run app/dashboard.py             # choose "Real USGS footage" to replay a held-out event
```

## Data

All data is public domain and needs no account or key. The repository commits the split files, the gauge labels (`data/usgs/truth/*.csv`) and the 68-frame demo sample. The full frames (about 8,450 images, roughly 2 GB) are downloaded by script and are gitignored:

```bash
python scripts/build_usgs_dataset.py --sites in_overland hancock in_north oh_wetland wi_mt1
```

More detail is in `data/usgs/README.md` and `docs/DATA_SOURCES.md`.

## Rerun the experiments

```bash
make real-strips      # SIFT registration and plate strips (about 1-2 h on 4 cores)
make real-eval        # CV per camera, resolution sweep, transfer to hancock, leave-one-camera-out, report
make paper            # regenerate paper numbers, tables and figures, then build main.pdf
```

| Result | Script | Output |
|---|---|---|
| Main table: leave-one-month-out CV, baselines | `scripts/real_cv.py` | `results/real/real-cv-in_overland-f1/` |
| Locked test months | `scripts/real_final_test.py` | `results/real/real-final-in_overland-f1-force/`, `final_test_log.jsonl` |
| Per-camera table, in-site | `scripts/real_cv.py --site <camera>` | `results/real/real-cv-<camera>-f1/` |
| Per-camera table, camera held out | `scripts/real_multicam.py` | `results/real/loco/` |
| River camera | `scripts/real_transfer.py` | `results/real/transfer-in_overland-to-hancock/` |
| Resolution sweep and error model | `scripts/real_resolution.py` | `results/real/resolution-in_overland/`, `real-res-*` |
| Figures, tables, failure audit | `scripts/real_report.py`, `scripts/make_paper_assets.py` | `results/real/figures/`, `tables/`, `failures/`, `paper/` |
| This website's data and images | `scripts/build_site_assets.py` | `site/_data/`, `site/assets/` |

Each run writes a `meta.json` with the git commit, seed, data-manifest hash and split hash.

## Rules built into the code

- Splits are by calendar month, never by frame. They are committed; do not regenerate them.
- The locked test months were fixed before training. The final-test script refuses a second run without `--force` and logs every run.
- Everything learned is fitted on training months only. Seeds are fixed (`seed=0`).

## This website

The site is a Jekyll site using the [Just the Docs](https://just-the-docs.com/) theme. Its source is in `site/`, and a GitHub Actions workflow (`.github/workflows/pages.yml`) builds and deploys it when `site/` changes on `main`.

Numbers, tables and figures on the site are generated, not typed:

```bash
python scripts/build_site_assets.py     # rebuild site/_data and site/assets from results/ and paper/
```

The script needs `pdftoppm` (poppler or MiKTeX) to turn the paper figures into PNG. The built files are committed, so the CI build needs only Ruby.

To preview locally, with Ruby 3.3 and Bundler installed:

```bash
cd site
bundle install
bundle exec jekyll serve --livereload      # open http://localhost:4000/FloodCam/
```

Without Ruby, the same preview runs in Docker:

```bash
docker run --rm -p 4000:4000 -v "$PWD/site:/srv/jekyll" -w /srv/jekyll ruby:3.3 \
  sh -c "bundle install && bundle exec jekyll serve --host 0.0.0.0"
```

The demo is static: it replays stored held-out estimates next to the USGS gauge height, and it needs no server.
