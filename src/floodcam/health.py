"""Frame-level health checks for the FloodCam runner.

All checks operate on uint8 RGB frames. Thresholds come from HealthConfig.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from floodcam.config import HealthConfig


@dataclass
class HealthResult:
    frozen: bool
    black: bool
    obstructed: bool
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return not (self.frozen or self.black or self.obstructed)


class HealthChecker:
    """Stateful frame-level health checker for a single camera stream."""

    def __init__(self, cfg: HealthConfig) -> None:
        self._cfg = cfg
        self._prev_frames: deque[np.ndarray] = deque(maxlen=cfg.frozen_k_frames)
        self._laplacian_baseline: deque[float] = deque(
            maxlen=cfg.obstruction_laplacian_baseline_frames
        )
        self._frozen_count: int = 0

    def check(self, frame_rgb: np.ndarray) -> HealthResult:
        import cv2

        details: list[str] = []

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        luminance = float(np.mean(gray))

        # Black frame check
        black = luminance < self._cfg.black_luminance_threshold
        if black:
            details.append(f"black(lum={luminance:.1f})")

        # Frozen frame check
        frozen = False
        if len(self._prev_frames) >= 1:
            prev = self._prev_frames[-1]
            diff = float(
                np.mean(np.abs(frame_rgb.astype(np.int32) - prev.astype(np.int32)))
            )
            if diff < self._cfg.frozen_diff_threshold:
                self._frozen_count += 1
            else:
                self._frozen_count = 0
            if self._frozen_count >= self._cfg.frozen_k_frames - 1:
                frozen = True
                details.append(
                    f"frozen(diff={diff:.2f},count={self._frozen_count})"
                )
        self._prev_frames.append(frame_rgb.copy())

        # Obstruction check: Laplacian variance collapse
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self._laplacian_baseline.append(lap_var)
        obstructed = False
        if (
            len(self._laplacian_baseline)
            >= self._cfg.obstruction_laplacian_baseline_frames
        ):
            baseline_vals = list(self._laplacian_baseline)[:-1]
            baseline = float(np.median(baseline_vals))
            if baseline > 0 and lap_var < self._cfg.obstruction_laplacian_ratio * baseline:
                obstructed = True
                details.append(
                    f"obstructed(lap={lap_var:.1f},baseline={baseline:.1f})"
                )

        return HealthResult(
            frozen=frozen,
            black=black,
            obstructed=obstructed,
            detail="; ".join(details),
        )
