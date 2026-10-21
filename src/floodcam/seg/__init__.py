"""Water segmentation: HSV baseline and backend protocol."""

from floodcam.seg.backend import HsvBackend, SegmentationBackend
from floodcam.seg.hsv import apply_hsv_mask

__all__ = ["apply_hsv_mask", "HsvBackend", "SegmentationBackend"]
