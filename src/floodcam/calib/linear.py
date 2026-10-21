"""Linear scale calibration: OLS fit of height_cm = a * pixel_row + b.

Calibration data-flow contract
-------------------------------
fit() accepts only explicit (pixel_row, height_cm) pairs from reference marks.
It never reads frame images, depth labels, or evaluation metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from floodcam.config import CalibrationConfig


class CalibError(ValueError):
    """Invalid calibration input or attempt to use an unfitted calibrator."""


@dataclass
class LinearCalib:
    """Least-squares linear fit: height_cm = a * pixel_row + b.

    Sign convention
    ---------------
    For a downward-looking camera, higher pixel rows (larger v) correspond to
    lower heights. Therefore a < 0 (slope is negative) and scale = |a| > 0.

    sigma_s definition
    ------------------
    sigma_s is the OLS standard error of the slope a:

        MSE  = sum(residuals^2) / (n - 2)      for n > 2, else 0
        Sxx  = sum((v_i - v_mean)^2)
        sigma_s = sqrt(MSE / Sxx)

    This is the 1-sigma uncertainty on the cm/pixel scale factor |a|.
    sigma_s = 0 when n == 2 (exact fit, no residual degrees of freedom).
    """

    a: Optional[float] = None
    b: Optional[float] = None
    rms_residual: Optional[float] = None
    sigma_s: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, points: list[tuple[float, float]]) -> "LinearCalib":
        """Fit h = a*v + b from (pixel_row, height_cm) pairs.

        Args:
            points: List of (pixel_row, height_cm). Minimum 2 required.

        Returns:
            self (allows chaining: LinearCalib().fit(pts)).

        Raises:
            CalibError: fewer than 2 points, or all rows identical.
        """
        if len(points) < 2:
            raise CalibError(
                f"LinearCalib.fit() needs >= 2 calibration points; got {len(points)}."
            )

        vs = np.array([p[0] for p in points], dtype=np.float64)
        hs = np.array([p[1] for p in points], dtype=np.float64)

        if np.allclose(vs, vs[0]):
            raise CalibError(
                "All calibration points have identical pixel rows; cannot determine scale."
            )

        n = len(points)
        A = np.stack([vs, np.ones(n, dtype=np.float64)], axis=1)
        params, _, _, _ = np.linalg.lstsq(A, hs, rcond=None)
        self.a = float(params[0])
        self.b = float(params[1])

        residuals = hs - (A @ params)
        self.rms_residual = float(np.sqrt(np.mean(residuals**2)))

        if n > 2:
            mse = float(np.sum(residuals**2) / (n - 2))
            sxx = float(np.sum((vs - vs.mean()) ** 2))
            self.sigma_s = float(math.sqrt(mse / sxx)) if sxx > 0 else 0.0
        else:
            self.sigma_s = 0.0

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def height_at(self, v: float) -> float:
        """Estimated height in cm at pixel row v."""
        self._require_fitted()
        return float(self.a * v + self.b)

    def cm_per_pixel(self) -> float:
        """Scale factor |dh/dv| in cm/pixel (constant for linear model)."""
        self._require_fitted()
        return abs(self.a)

    def cm_per_pixel_at(self, v: float) -> float:
        """Scale at pixel row v — same as cm_per_pixel() for a linear model."""
        return self.cm_per_pixel()

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_config_updates(self) -> dict:
        """Dict of CalibrationConfig fields set by this fit result."""
        self._require_fitted()
        return {
            "scale_cm_per_px": abs(self.a),
            "intercept_cm": self.b,
            "sigma_s": self.sigma_s,
        }

    @classmethod
    def from_config(cls, cfg: "CalibrationConfig") -> "LinearCalib":
        """Reconstruct a fitted LinearCalib from a loaded CalibrationConfig.

        Requires cfg.scale_cm_per_px and cfg.intercept_cm to be non-None.
        scale_cm_per_px = |a|; for a standard downward-looking camera a = -scale.
        """
        if cfg.scale_cm_per_px is None or cfg.intercept_cm is None:
            raise CalibError(
                "Cannot reconstruct LinearCalib: scale_cm_per_px or intercept_cm is None "
                "in CalibrationConfig."
            )
        obj = cls()
        obj.a = -abs(float(cfg.scale_cm_per_px))
        obj.b = float(cfg.intercept_cm)
        obj.sigma_s = float(cfg.sigma_s) if cfg.sigma_s is not None else 0.0
        obj.rms_residual = None
        return obj

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if self.a is None:
            raise CalibError("LinearCalib is not fitted; call fit() first.")
