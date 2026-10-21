"""Generate train/val/test splits from a scene manifest.

Splits by scene/session (never by frame). Outputs one JSON file per split
to data/splits/. Run once; committed split files are authoritative and must
not be regenerated without --force.

Usage:
    python scripts/generate_splits.py --manifest data/manifest.json --seed 42
"""

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Scene manifest JSON (list of {scene_id, ...})"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-frac", type=float, default=0.15, help="Fraction of scenes for validation"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.15, help="Fraction of scenes for test (locked)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files")
    args = parser.parse_args()

    existing = list(SPLITS_DIR.glob("*.json"))
    if existing and not args.force:
        print(
            f"ERROR: Split files already exist in {SPLITS_DIR}.\n"
            "       Use --force to regenerate (this resets all splits).",
            file=sys.stderr,
        )
        return 1

    with args.manifest.open() as f:
        manifest = json.load(f)

    scenes = [s["scene_id"] for s in manifest["scenes"]]
    if not scenes:
        print("ERROR: Manifest contains no scenes.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shuffled = scenes.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * args.test_frac))
    n_val = max(1, round(n * args.val_frac))
    n_train = n - n_val - n_test

    if n_train < 1:
        print(
            f"ERROR: {n} scenes is too few for a train/val/test split "
            f"(val_frac={args.val_frac}, test_frac={args.test_frac}).",
            file=sys.stderr,
        )
        return 1

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    # Verify no leakage before writing
    all_sets = [set(train_ids), set(val_ids), set(test_ids)]
    for i, a in enumerate(all_sets):
        for j, b in enumerate(all_sets):
            if i >= j:
                continue
            overlap = a & b
            if overlap:
                print(f"BUG: overlap between split {i} and {j}: {overlap}", file=sys.stderr)
                return 1

    today = date.today().isoformat()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out = {
            "seed": args.seed,
            "created": today,
            "split": split_name,
            "scene_ids": ids,
        }
        p = SPLITS_DIR / f"{split_name}.json"
        with p.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"  {split_name}: {len(ids)} scenes -> {p}")

    print(
        f"\nSplit complete (seed={args.seed}): "
        f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
