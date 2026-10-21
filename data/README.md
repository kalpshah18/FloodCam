# Data Directory

This directory is git-ignored except for `splits/` and this README.

## Public segmentation datasets (do not commit)

| Dataset | License | Notes |
|---------|---------|-------|
| ADE20K | CC BY 4.0 | Indoor/outdoor scenes with water labels |
| COCO-Stuff | CC BY 4.0 | Broad scene categories including water |
| FloodNet | CC BY-SA 4.0 | Aerial flood imagery |

Download to `data/raw/<dataset_name>/`. Check each license before use.

## Self-collected depth dataset (human task)

See `docs/RECORDING_PROTOCOL.md` for exact recording steps.
Place recordings under `data/raw/self_collected/<session_id>/`.

## Directory layout

```
data/
├── raw/                      # downloaded/recorded data (gitignored)
│   ├── ade20k/
│   ├── coco_stuff/
│   ├── floodnet/
│   └── self_collected/
├── processed/                # preprocessed frames + masks (gitignored)
├── splits/                   # train/val/test scene IDs (committed)
│   ├── train.json
│   ├── val.json
│   └── test.json
└── README.md
```

## Generating splits

Once you have a scene manifest at `data/manifest.json`:

```bash
python scripts/generate_splits.py --manifest data/manifest.json --seed 42
```

Committed split files in `data/splits/` are authoritative. Never regenerate
without `--force`, and document the reason in `docs/EXPERIMENT_LOG.md`.
