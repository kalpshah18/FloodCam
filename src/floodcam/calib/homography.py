"""Homography calibration via normalized DLT.

Maps image coordinates (u, v) to plane coordinates (x_cm, height_cm) where the
plane is the vertical surface of the reference object (pole or wall).

Calibration data-flow contract
-------------------------------
fit() accepts only explicit (u, v, x_cm, height_cm) correspondences from
reference marks. No frame images or evaluation labels are read inside fit().
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from floodcam.calib.linear import CalibError

if TYPE_CHECKING:
    from floodcam.config import CalibrationConfig

# Threshold for collinearity / degenerate-config detection
_COLLINEAR_RATIO = 1e-7


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Isotropic normalization T so that normalized pts have centroid 0 and
    mean distance sqrt(2) from origin."""
    cx, cy = pts.mean(axis=0)
    dists = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1))
    mean_dist = float(dists.mean())
    if mean_dist < 1e-10:
        raise CalibError("Calibration points are coincident; cannot build normalization matrix.")
    scale = math.sqrt(2) / mean_dist
    return np.array(
        [[scale, 0.0, -scale * cx], [0.0, scale, -scale * cy], [0.0, 0.0, 1.0]]
    )


def _check_not_collinear(pts: np.ndarray, label: str) -> None:
    """Raise CalibError if 2-D points are approximately collinear."""
    centered = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    if sv[-1] < _COLLINEAR_RATIO * sv[0]:
        raise CalibError(f"{label} calibration points are collinear; homography is undefined.")


def _check_no_duplicates(pts: np.ndarray, label: str) -> None:
    norms = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(norms, np.inf)
    if np.any(norms < 1e-9):
        raise CalibError(f"{label} calibration points contain duplicates.")


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute H (3×3) such that dst ≈ H @ src in homogeneous coordinates.

    src: (N, 2) world/plane points (x_cm, y_cm)
    dst: (N, 2) image points (u, v)

    Returns H normalized so H[2,2] != 0.
    """
    N = len(src)
    T_src = _normalization_matrix(src)
    T_dst = _normalization_matrix(dst)

    src_h = np.column_stack([src, np.ones(N)])
    dst_h = np.column_stack([dst, np.ones(N)])
    src_n = (T_src @ src_h.T).T
    dst_n = (T_dst @ dst_h.T).T

    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = src_n[i, 0], src_n[i, 1]
        u, v = dst_n[i, 0], dst_n[i, 1]
        A[2 * i]     = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1].reshape(3, 3)

    T_dst_inv = np.linalg.inv(T_dst)
    H = T_dst_inv @ h @ T_src
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H


class HomographyCalib:
    """Normalized-DLT homography: image (u,v) → plane (x_cm, height_cm).

    H maps plane to image: λ [u, v, 1]ᵀ = H [x, y, 1]ᵀ.
    height_at() and cm_per_pixel_at() use H⁻¹ to go image → world.
    """

    def __init__(self) -> None:
        self._H: Optional[np.ndarray] = None
        self._H_inv: Optional[np.ndarray] = None
        self.reprojection_rms: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, points: list[tuple[float, float, float, float]]) -> "HomographyCalib":
        """Fit homography from >= 4 non-collinear 2D correspondences.

        Args:
            points: List of (image_u, image_v, plane_x_cm, plane_height_cm).
                    Minimum 4 non-collinear, non-duplicate points in both spaces.

        Returns:
            self (allows chaining).

        Raises:
            CalibError: < 4 points, duplicates, or collinear configuration.
        """
        if len(points) < 4:
            raise CalibError(
                f"HomographyCalib.fit() needs >= 4 correspondences; got {len(points)}."
            )

        img_pts = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
        world_pts = np.array([[p[2], p[3]] for p in points], dtype=np.float64)

        _check_no_duplicates(img_pts, "Image")
        _check_no_duplicates(world_pts, "World")
        _check_not_collinear(img_pts, "Image")
        _check_not_collinear(world_pts, "World")

        # H maps world -> image
        self._H = _dlt_homography(world_pts, img_pts)
        self._H_inv = np.linalg.inv(self._H)
        self.reprojection_rms = self._compute_reprojection_rms(world_pts, img_pts)
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def height_at(self, u: float, v: float) -> float:
        """Estimated height in cm at image position (u, v)."""
        self._require_fitted()
        p = self._H_inv @ np.array([u, v, 1.0], dtype=np.float64)
        return float(p[1] / p[2])

    def cm_per_pixel_at(self, u: float, v: float) -> float:
        """Scale |dh/dv| in cm/pixel at image position (u, v)."""
        self._require_fitted()
        H = self._H_inv
        w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        num = H[1, 0] * u + H[1, 1] * v + H[1, 2]
        dydv = (H[1, 1] * w - num * H[2, 1]) / (w ** 2)
        return abs(float(dydv))

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_config_updates(self) -> dict:
        """Dict of CalibrationConfig fields set by this fit result."""
        self._require_fitted()
        return {"homography_matrix": self._H.tolist()}

    @classmethod
    def from_config(cls, cfg: "CalibrationConfig") -> "HomographyCalib":
        """Reconstruct a fitted HomographyCalib from a CalibrationConfig."""
        if cfg.homography_matrix is None:
            raise CalibError(
                "Cannot reconstruct HomographyCalib: homography_matrix is None."
            )
        obj = cls()
        obj._H = np.array(cfg.homography_matrix, dtype=np.float64)
        obj._H_inv = np.linalg.inv(obj._H)
        obj.reprojection_rms = None
        return obj

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _compute_reprojection_rms(
        self, world_pts: np.ndarray, img_pts: np.ndarray
    ) -> float:
        N = len(world_pts)
        src_h = np.column_stack([world_pts, np.ones(N)])
        proj = (self._H @ src_h.T).T
        proj /= proj[:, 2:3]
        errors = np.sqrt(np.sum((proj[:, :2] - img_pts) ** 2, axis=1))
        return float(np.sqrt(np.mean(errors ** 2)))

    def _require_fitted(self) -> None:
        if self._H is None:
            raise CalibError("HomographyCalib is not fitted; call fit() first.")
