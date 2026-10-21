"""Tests for the config loader (src/floodcam/config.py)."""
from pathlib import Path

import pytest
import yaml

from floodcam.config import (
    CameraConfig,
    ExperimentConfig,
    load_camera_config,
    load_experiment_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.fps_sample == 1
        assert cfg.hysteresis.trigger_cm == 3.5
        assert cfg.hysteresis.clear_cm == 2.5
        assert cfg.severity.advisory_cm == 3.0
        assert cfg.severity.warning_cm == 10.0

    def test_from_dict_partial_preserves_defaults(self):
        cfg = CameraConfig.from_dict({"camera_id": "cam_x", "fps_sample": 3})
        assert cfg.camera_id == "cam_x"
        assert cfg.fps_sample == 3
        assert cfg.hysteresis.trigger_cm == 3.5  # default preserved

    def test_nested_roi_from_dict(self):
        cfg = CameraConfig.from_dict({"roi": {"x": 10, "y": 20, "w": 320, "h": 240}})
        assert cfg.roi.x == 10
        assert cfg.roi.h == 240

    def test_load_from_yaml(self, tmp_path):
        data = {"camera_id": "cam_yaml", "fps_sample": 5}
        p = tmp_path / "cam.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_camera_config(p)
        assert cfg.camera_id == "cam_yaml"
        assert cfg.fps_sample == 5

    def test_config_hash_deterministic(self):
        cfg = CameraConfig()
        assert cfg.config_hash() == cfg.config_hash()

    def test_config_hash_sensitive_to_values(self):
        cfg1 = CameraConfig(fps_sample=1)
        cfg2 = CameraConfig(fps_sample=2)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_camera_template_yaml_loads(self):
        p = CONFIGS_DIR / "camera_template.yaml"
        if not p.exists():
            pytest.skip("camera_template.yaml not found")
        cfg = load_camera_config(p)
        assert cfg.camera_id == "cam_template"


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.seed == 42
        assert cfg.model == "hsv_baseline"
        assert cfg.hsv_lower == [90, 50, 50]
        assert cfg.hsv_upper == [130, 255, 255]

    def test_load_from_yaml(self, tmp_path):
        data = {"name": "exp_test", "seed": 7, "model": "segformer_b0"}
        p = tmp_path / "exp.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_experiment_config(p)
        assert cfg.name == "exp_test"
        assert cfg.seed == 7
        assert cfg.model == "segformer_b0"

    def test_unknown_keys_ignored(self, tmp_path):
        data = {"name": "x", "nonexistent_key": 999}
        p = tmp_path / "exp.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_experiment_config(p)
        assert cfg.name == "x"

    def test_default_yaml_valid(self):
        p = CONFIGS_DIR / "default.yaml"
        if not p.exists():
            pytest.skip("default.yaml not found")
        cfg = load_experiment_config(p)
        assert cfg.name == "default"
