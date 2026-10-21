"""Camera calibration: linear scale fitting and homography.

Public API:
    LinearCalib        — OLS fit of h = a*v + b
    HomographyCalib    — Normalized DLT 2D correspondence
    depth_uncertainty_cm — Propagate waterline + scale errors to depth error
    CalibError         — Raised on invalid input or unfitted usage
    annotate_headless  — Headless annotation CLI helper
"""

from floodcam.calib.homography import HomographyCalib
from floodcam.calib.linear import CalibError, LinearCalib
from floodcam.calib.uncertainty import depth_uncertainty_cm

__all__ = [
    "CalibError",
    "LinearCalib",
    "HomographyCalib",
    "depth_uncertainty_cm",
]
