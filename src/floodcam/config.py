"""Config dataclasses for cameras and experiments. Loaded from YAML files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SeverityThresholds:
    advisory_cm: float = 3.0
    warning_cm: float = 10.0
    severe_cm: float = 25.0
    critical_cm: float = 50.0
    rate_escalation_cm_per_10min: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> SeverityThresholds:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HysteresisConfig:
    trigger_cm: float = 3.5
    clear_cm: float = 2.5
    persistence_min: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> HysteresisConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CalibrationConfig:
    mode: str = "linear"  # "linear" | "homography"
    scale_cm_per_px: Optional[float] = None
    intercept_cm: Optional[float] = None       # b in h = a*v + b
    sigma_s: Optional[float] = None            # OLS std error of slope (cm/px)
    reference_points: list[list[float]] = field(default_factory=list)
    homography_matrix: Optional[list[list[float]]] = None

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReferenceObject:
    """Known-size object (pole, curb) used as the geometric reference."""

    known_height_cm: float = 100.0
    pole_x_frac: float = 0.5  # fractional column position in frame [0, 1]
    pole_width_px: int = 6

    @classmethod
    def from_dict(cls, d: dict) -> ReferenceObject:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ROI:
    """Rectangular region of interest in pixel coordinates."""

    x: int = 0
    y: int = 0
    w: int = 640
    h: int = 480

    @classmethod
    def from_dict(cls, d: dict) -> ROI:
        return cls(**{k: int(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TemporalConfig:
    """Temporal filter and state machine parameters for one camera."""

    filter_window_samples: int = 7
    min_confidence: float = 0.5
    persistence_minutes: float = 5.0
    trend_window_minutes: float = 10.0
    falling_slope_cm_per_10min: float = -1.0
    rise_escalation_cm_per_10min: float = 5.0
    trigger_cm: float = 3.5
    clear_cm: float = 2.5
    max_gap_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not (5 <= self.filter_window_samples <= 10):
            raise ValueError(
                f"filter_window_samples must be in [5, 10], got {self.filter_window_samples}"
            )
        if self.trigger_cm <= self.clear_cm:
            raise ValueError(
                f"trigger_cm ({self.trigger_cm}) must be > clear_cm ({self.clear_cm})"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HealthConfig:
    """Thresholds for frame-level health checks."""

    frozen_diff_threshold: float = 5.0
    frozen_k_frames: int = 5  # raised from 3: JPEG compression can produce sporadic sub-threshold diffs
    black_luminance_threshold: float = 10.0
    obstruction_laplacian_baseline_frames: int = 30
    obstruction_laplacian_ratio: float = 0.3
    offline_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.frozen_diff_threshold < 0:
            raise ValueError("frozen_diff_threshold must be non-negative")

    @classmethod
    def from_dict(cls, d: dict) -> "HealthConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CameraConfig:
    camera_id: str = "cam_default"
    description: str = ""
    fps_sample: int = 1
    roi: ROI = field(default_factory=ROI)
    reference_object: ReferenceObject = field(default_factory=ReferenceObject)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    severity: SeverityThresholds = field(default_factory=SeverityThresholds)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    @classmethod
    def from_dict(cls, d: dict) -> CameraConfig:
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k == "roi":
                kwargs[k] = ROI.from_dict(v)
            elif k == "reference_object":
                kwargs[k] = ReferenceObject.from_dict(v)
            elif k == "calibration":
                kwargs[k] = CalibrationConfig.from_dict(v)
            elif k == "severity":
                kwargs[k] = SeverityThresholds.from_dict(v)
            elif k == "hysteresis":
                kwargs[k] = HysteresisConfig.from_dict(v)
            elif k == "temporal":
                kwargs[k] = TemporalConfig.from_dict(v)
            elif k == "health":
                kwargs[k] = HealthConfig.from_dict(v)
            elif k in cls.__dataclass_fields__:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path | str) -> CameraConfig:
        with Path(path).open() as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def save(self, path: Path | str) -> None:
        with Path(path).open("w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=True)

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ExperimentConfig:
    name: str = "default"
    model: str = "hsv_baseline"  # "hsv_baseline"|"segformer_b0"|"segformer_b1"|"cnn_classifier"
    seed: int = 42
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hsv_lower: list[int] = field(default_factory=lambda: [90, 50, 50])
    hsv_upper: list[int] = field(default_factory=lambda: [130, 255, 255])
    hsv_morph_open: int = 0
    hsv_morph_close: int = 0
    runner_sample_fps: float = 1.0
    image_size: int = 512
    num_workers: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path | str) -> ExperimentConfig:
        with Path(path).open() as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_camera_config(path: Path | str) -> CameraConfig:
    return CameraConfig.load(path)


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    return ExperimentConfig.load(path)
