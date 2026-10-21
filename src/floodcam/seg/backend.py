"""Segmentation backend protocol and HSV implementation.

The runner depends only on the SegmentationBackend Protocol so that SegFormer
can be swapped in at Milestone 8 without changing the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from floodcam.config import CameraConfig, ExperimentConfig


class SegmentationBackend(Protocol):
    """Protocol for segmentation backends."""

    model_version: str

    def predict_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return (H, W) uint8 binary mask (0 or 255)."""
        ...


class HsvBackend:
    """SegmentationBackend wrapping apply_hsv_mask."""

    def __init__(
        self,
        exp_cfg: "ExperimentConfig",
        camera_cfg: "CameraConfig | None" = None,
    ) -> None:
        self._exp_cfg = exp_cfg
        self._camera_cfg = camera_cfg
        self.model_version = "hsv_baseline_v1"

    def predict_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        from floodcam.seg.hsv import apply_hsv_mask

        roi = self._camera_cfg.roi if self._camera_cfg is not None else None
        return apply_hsv_mask(
            frame_rgb,
            tuple(self._exp_cfg.hsv_lower),  # type: ignore[arg-type]
            tuple(self._exp_cfg.hsv_upper),   # type: ignore[arg-type]
            roi=roi,
            morph_open=self._exp_cfg.hsv_morph_open,
            morph_close=self._exp_cfg.hsv_morph_close,
        )
