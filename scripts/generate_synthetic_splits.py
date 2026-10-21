"""Generate scene-level splits for the synthetic dataset.

Produces:
  - train / val / test scene partition (coarse split)
  - leave-one-scene-out (LOSO) folds over all non-locked, non-negative scenes
  - one locked final-test scene held out from ALL folds

Output: data/splits/synthetic_splits.json (never overwritten without --force).
This file is separate from real-data splits (train.json / val.json / test.json).

Usage:
    python scripts/generate_synthetic_splits.py \
        --manifest data/synthetic/manifest.jsonl \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
SYNTH_SPLITS_FILE = ROOT / "data" / "splits" / "synthetic_splits.json"


def _load_scene_ids(manifest_path: Path, exclude_negative: bool = True) -> list[str]:
    """Return ordered unique scene IDs from a JSONL manifest."""
    seen: dict[str, None] = {}  # ordered set
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if exclude_negative and row.get("is_negative", False):
                continue
            scene_id = row["scene_id"]
            if scene_id not in seen:
                seen[scene_id] = None
    return list(seen)


def _make_loso_folds(
    scenes: list[str],
) -> list[dict]:
    """Return LOSO folds: each fold holds out one scene for validation."""
    folds = []
    for i, val_scene in enumerate(scenes):
        train_scenes = [s for s in scenes if s != val_scene]
        folds.append(
            {
                "fold_idx": i,
                "val_scene": val_scene,
                "train_scenes": train_scenes,
            }
        )
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Synthetic manifest JSONL path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing synthetic_splits.json"
    )
    args = parser.parse_args()

    if SYNTH_SPLITS_FILE.exists() and not args.force:
        print(
            f"ERROR: {SYNTH_SPLITS_FILE} already exists. Use --force to regenerate.",
            file=sys.stderr,
        )
        return 1

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    scene_ids = _load_scene_ids(args.manifest, exclude_negative=True)
    if len(scene_ids) < 3:
        print(
            f"ERROR: Need at least 3 non-negative scenes; found {len(scene_ids)}.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    shuffled = scene_ids.copy()
    rng.shuffle(shuffled)

    # Lock the last scene as final test
    locked_test = shuffled[-1]
    loso_scenes = shuffled[:-1]  # everything except locked test

    # Also produce a coarse train/val partition of the loso scenes
    n = len(loso_scenes)
    n_val = max(1, round(n * 0.20))
    coarse_val = loso_scenes[:n_val]
    coarse_train = loso_scenes[n_val:]

    loso_folds = _make_loso_folds(loso_scenes)

    # Sanity: locked test must not appear in any fold
    for fold in loso_folds:
        assert locked_test not in fold["train_scenes"]
        assert fold["val_scene"] != locked_test

    out = {
        "seed": args.seed,
        "created": date.today().isoformat(),
        "type": "synthetic_loso",
        "locked_test_scene": locked_test,
        "coarse_train_scenes": coarse_train,
        "coarse_val_scenes": coarse_val,
        "loso_folds": loso_folds,
    }

    SYNTH_SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SYNTH_SPLITS_FILE.open("w") as fh:
        json.dump(out, fh, indent=2)

    print(f"Synthetic splits written to {SYNTH_SPLITS_FILE}")
    print(f"  Locked test scene : {locked_test}")
    print(f"  LOSO scenes       : {len(loso_scenes)}")
    print(f"  LOSO folds        : {len(loso_folds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
