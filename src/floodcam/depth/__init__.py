"""Depth estimation: waterline detection and calibrated depth conversion."""

from floodcam.depth.estimate import DepthEstimate, estimate, make_camera_config_for_renderer
from floodcam.depth.waterline import WaterlineResult, detect_waterline

__all__ = [
    "WaterlineResult",
    "detect_waterline",
    "DepthEstimate",
    "estimate",
    "make_camera_config_for_renderer",
]
