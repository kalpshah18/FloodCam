#!/usr/bin/env python
"""Train SegFormer. Thin wrapper around floodcam.seg.train.main.

  python scripts/train_seg.py --config configs/seg_pretrain.yaml --run-id seg-pre-001
  python scripts/train_seg.py --config configs/seg_finetune.yaml --run-id seg-ft-001
  python scripts/train_seg.py --smoke        # tiny random model, no downloads, 2 steps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floodcam.seg.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
