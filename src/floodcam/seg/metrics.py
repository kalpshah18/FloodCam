"""Binary-mask metrics accumulated over a whole split (not mean of per-image scores)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        p, t = pred.astype(bool), target.astype(bool)
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())
        self.tn += int((~p & ~t).sum())

    def iou(self) -> Optional[float]:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else None

    def f1(self) -> Optional[float]:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else None

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "iou": self.iou(), "f1": self.f1(),
        }
