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
class CameraConfig:
    camera_id: str = "cam_default"
    description: str = ""
    fps_sample: int = 1
    roi: ROI = field(default_factory=ROI)
    reference_object: ReferenceObject = field(default_factory=ReferenceObject)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    severity: SeverityThresholds = field(default_factory=SeverityThresholds)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)

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
            elif k in cls.__dataclass_fields__:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path | str) -> CameraConfig:
        with Path(path).open() as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

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
