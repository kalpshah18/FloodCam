"""Interactive pipeline for user-supplied videos (dashboard upload mode).

Pipeline per sampled frame:
    health -> segmentation backend -> waterline -> calibrated depth
    -> temporal filter -> state machine -> (optional) SQLite event log

Calibration for an arbitrary video is entered by the user as two marks on the
reference pole: the ground-contact row (height 0) and a second row at a known
height. That is a 2-point LinearCalib (h = a*v + b); see docs/CALIBRATION.md.
Use the full homography flow (floodcam.calib.annotate) when camera tilt is large.

Timestamps are video time, so results are independent of decode/replay speed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

from floodcam.calib.linear import CalibError, LinearCalib
from floodcam.config import (
    ROI,
    CameraConfig,
    ExperimentConfig,
    HealthConfig,
    TemporalConfig,
)
from floodcam.depth.estimate import DepthEstimate, estimate_from_mask
from floodcam.health import HealthChecker
from floodcam.io.stabilise import Stabiliser
from floodcam.io.video import DEFAULT_MAX_WIDTH, iter_sampled_frames, probe_video, read_reference
from floodcam.temporal.filter import TemporalFilter
from floodcam.temporal.state_machine import StateMachine

if TYPE_CHECKING:
    from floodcam.seg.backend import SegmentationBackend
    from floodcam.store.db import EventStore

DRY_CONFIDENCE = 0.8  # confidence assigned to a healthy 'no water at pole' reading

# Named HSV presets (OpenCV ranges: H 0-179). Starting points only; tune on the preview.
HSV_PRESETS: dict[str, tuple[list[int], list[int]]] = {
    "Clear / blue-green water": ([85, 40, 40], [130, 255, 255]),
    "Muddy / brown water": ([8, 40, 40], [30, 255, 210]),
    "Grey / overcast water": ([0, 0, 60], [179, 45, 200]),
}


@dataclass
class LiveSettings:
    """Everything the user configures for one uploaded video."""

    # Reference-pole calibration, in processed-frame pixels
    pole_x_px: int
    ground_row: float          # image row where the pole meets the ground (h = 0)
    mark_row: float            # image row of a second mark on the pole
    mark_height_cm: float      # real height of that mark above ground
    pole_width_px: int = 6
    roi: Optional[tuple[int, int, int, int]] = None  # x, y, w, h; None = full frame
    # Segmentation / depth reading
    mask_source: str = "change"   # "change" (vs dry reference) | "colour" (HSV) | "model" (SegFormer)
    estimator: str = "gauge"      # "gauge" (on the pole strip) | "flank" (water beside the pole)
    diff_threshold: float = 20.0  # Lab dE for the change mask
    diff_blur: int = 7
    ref_time_s: float = 0.0       # where the dry reference starts
    stabilise: bool = True
    hsv_lower: list[int] = field(default_factory=lambda: [85, 40, 40])
    hsv_upper: list[int] = field(default_factory=lambda: [130, 255, 255])
    morph_open: int = 3
    morph_close: int = 5
    # Sampling
    sample_fps: float = 1.0
    max_width: int = DEFAULT_MAX_WIDTH
    # Temporal logic (seconds; short-clip friendly)
    filter_window: int = 5
    min_confidence: float = 0.3
    persistence_s: float = 10.0
    trend_window_s: float = 20.0
    trigger_cm: float = 3.5
    clear_cm: float = 2.5
    rate_escalation: bool = False


def build_camera_config(
    settings: LiveSettings, frame_shape: tuple[int, ...], camera_id: str = "upload"
) -> CameraConfig:
    """Fit the 2-point linear calibration and assemble a CameraConfig.

    Raises CalibError for geometry that cannot be calibrated (e.g. the marked
    height is not above the ground row, or zero height).
    """
    H, W = frame_shape[:2]
    if settings.mark_height_cm <= 0:
        raise CalibError("Mark height must be > 0 cm.")
    if settings.mark_row >= settings.ground_row:
        raise CalibError("The marked point must be above the ground-contact row (smaller row).")
    if not (0 <= settings.mark_row and settings.ground_row <= H - 1):
        raise CalibError("Calibration rows must lie inside the frame.")
    if not (0 <= settings.pole_x_px < W):
        raise CalibError("Pole column must lie inside the frame.")

    lc = LinearCalib().fit([(settings.ground_row, 0.0), (settings.mark_row, settings.mark_height_cm)])

    max_gap = max(30.0, 5.0 / max(settings.sample_fps, 1e-6))
    cfg = CameraConfig(camera_id=camera_id)
    cfg.description = "User-uploaded video (2-point linear calibration)"
    cfg.calibration.mode = "linear"
    upd = lc.to_config_updates()
    cfg.calibration.scale_cm_per_px = upd["scale_cm_per_px"]
    cfg.calibration.intercept_cm = upd["intercept_cm"]
    cfg.calibration.sigma_s = upd["sigma_s"]
    cfg.calibration.reference_points = [
        [float(settings.ground_row), 0.0],
        [float(settings.mark_row), float(settings.mark_height_cm)],
    ]
    cfg.reference_object.pole_x_frac = settings.pole_x_px / W
    cfg.reference_object.pole_width_px = settings.pole_width_px
    # Pole top is never visible in an arbitrary video; use the marked height + margin so
    # "fully submerged" fires only when water reaches above the highest calibrated mark.
    # Gauge mode extrapolates above the second mark (the pole top is unknown), so push the
    # "pole top" row far up; flank mode keeps the mark as the top.
    cfg.reference_object.known_height_cm = settings.mark_height_cm * (
        4.0 if settings.estimator == "gauge" else 1.0
    )
    x, y, w, h = settings.roi if settings.roi else (0, 0, W, H)
    cfg.roi = ROI(x=int(x), y=int(y), w=int(w), h=int(h))
    cfg.temporal = TemporalConfig(
        filter_window_samples=settings.filter_window,
        min_confidence=settings.min_confidence,
        persistence_minutes=settings.persistence_s / 60.0,
        trend_window_minutes=settings.trend_window_s / 60.0,
        trigger_cm=settings.trigger_cm,
        clear_cm=settings.clear_cm,
        max_gap_seconds=max_gap,
        rise_escalation_cm_per_10min=5.0 if settings.rate_escalation else 1e9,
    )
    cfg.health = HealthConfig(frozen_diff_threshold=0.5)
    return cfg


def build_experiment_config(settings: LiveSettings) -> ExperimentConfig:
    return ExperimentConfig(
        name="upload",
        hsv_lower=list(settings.hsv_lower),
        hsv_upper=list(settings.hsv_upper),
        hsv_morph_open=settings.morph_open,
        hsv_morph_close=settings.morph_close,
        runner_sample_fps=settings.sample_fps,
    )


def build_backend(
    settings: LiveSettings,
    cfg: CameraConfig,
    reference_rgb: Optional[np.ndarray],
    model_backend: Optional["SegmentationBackend"] = None,
) -> "SegmentationBackend":
    """Segmentation backend for the chosen mask source."""
    from floodcam.seg.backend import HsvBackend
    from floodcam.seg.change import ChangeBackend

    if model_backend is not None:
        return model_backend
    exp = build_experiment_config(settings)
    if settings.mask_source == "change":
        if reference_rgb is None:
            raise ValueError("The change mask needs a dry reference frame.")
        return ChangeBackend(
            reference_rgb, threshold=settings.diff_threshold, blur=settings.diff_blur,
            morph_open=settings.morph_open, morph_close=settings.morph_close,
            gain_rows=(0, max(8, int(settings.mark_row))),
        )
    return HsvBackend(exp, cfg)


@dataclass
class LiveFrame:
    """Result for one processed frame (all values are from the pipeline, none hand-set)."""

    index: int
    timestamp_s: float
    raw_depth_cm: Optional[float]      # per-frame estimate; None if invalid/unhealthy; 0.0 when dry
    depth_cm: float                    # temporally filtered depth
    confidence: float
    valid: bool
    reason: str
    state: str
    severity: str
    rate_cm_per_10min: Optional[float]
    transition: Optional[str]
    health_ok: bool
    health_detail: str
    annotated_rgb: np.ndarray
    frozen_flag: bool = False
    black_flag: bool = False
    obstructed_flag: bool = False


def annotate_frame(
    frame_rgb: np.ndarray,
    cfg: CameraConfig,
    mask: Optional[np.ndarray],
    waterline_row: Optional[float],
    depth_cm: Optional[float],
    state: str,
    show_mask: bool = True,
    severity: str = "",
) -> np.ndarray:
    """Overlay mask, gauge strip, ground line, waterline, depth bar and a readout. RGB out."""
    import cv2

    out = frame_rgb.copy()
    H, W = out.shape[:2]
    if show_mask and mask is not None and mask.any():
        tint = np.array([0, 140, 255], dtype=np.float32)
        sel = mask > 0
        out[sel] = (out[sel].astype(np.float32) * 0.6 + tint * 0.4).astype(np.uint8)

    r = cfg.roi
    if (r.w, r.h) != (W, H):
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w - 1, r.y + r.h - 1), (255, 255, 0), 1)
    pole_x = int(W * cfg.reference_object.pole_x_frac)
    hw = max(1, cfg.reference_object.pole_width_px // 2)
    pts = cfg.calibration.reference_points
    v_g = int(round(pts[0][0])) if pts else None
    v_m = int(round(pts[1][0])) if len(pts) > 1 else None
    if v_g is not None and v_m is not None:
        cv2.rectangle(out, (pole_x - hw, v_m), (pole_x + hw, v_g), (255, 255, 255), 1)
        cv2.line(out, (max(0, pole_x - 60), v_g), (min(W - 1, pole_x + 60), v_g), (0, 255, 0), 2)
        cv2.line(out, (max(0, pole_x - 60), v_m), (min(W - 1, pole_x + 60), v_m), (255, 0, 255), 1)
    if waterline_row is not None and 0 <= waterline_row < H:
        v = int(round(waterline_row))
        cv2.line(out, (0, v), (W - 1, v), (255, 60, 60), 2)

    # Banner
    label = "no reading" if depth_cm is None else f"{depth_cm:.1f} cm"
    sub = "  ".join(dict.fromkeys(x for x in (severity, state) if x))  # drop duplicate words
    cv2.rectangle(out, (0, 0), (min(W, 330), 44), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (150, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Depth bar (ground -> second mark) at the right edge
    if v_g is not None and v_m is not None and depth_cm is not None and pts:
        mark_cm = pts[1][1]
        x0 = W - 26
        cv2.rectangle(out, (x0, v_m), (x0 + 14, v_g), (255, 255, 255), 1)
        frac = float(np.clip(depth_cm / max(mark_cm, 1e-6), 0, 1))
        top = int(round(v_g - frac * (v_g - v_m)))
        cv2.rectangle(out, (x0 + 1, top), (x0 + 13, v_g - 1), (40, 140, 255), -1)
    return out


class LivePipeline:
    """Stateful per-frame pipeline. Feed frames in time order via step()."""

    def __init__(
        self,
        cfg: CameraConfig,
        exp_cfg: ExperimentConfig,
        backend: Optional["SegmentationBackend"] = None,
        estimator: str = "flank",
        reference_rgb: Optional[np.ndarray] = None,
        stabilise: bool = False,
        stab_rows: Optional[tuple[int, int]] = None,
    ) -> None:
        from floodcam.seg.backend import HsvBackend

        self.cfg = cfg
        self.exp_cfg = exp_cfg
        self.estimator = estimator
        self.backend = backend or HsvBackend(exp_cfg, cfg)
        self.stabiliser = (
            Stabiliser(reference_rgb, rows=stab_rows)
            if (stabilise and reference_rgb is not None) else None
        )
        self.health = HealthChecker(cfg.health)
        self.filter = TemporalFilter(cfg.temporal)
        self.machine = StateMachine(cfg.temporal)
        self._n = 0

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def _roi_masked(self, mask: np.ndarray) -> np.ndarray:
        r = self.cfg.roi
        out = np.zeros_like(mask)
        out[r.y : r.y + r.h, r.x : r.x + r.w] = mask[r.y : r.y + r.h, r.x : r.x + r.w]
        return out

    def step(self, timestamp_s: float, frame_rgb: np.ndarray) -> LiveFrame:
        if self.stabiliser is not None:
            frame_rgb = self.stabiliser.align(frame_rgb)
        hres = self.health.check(frame_rgb)
        # Frozen detection is informational only: a static fixed-camera scene legitimately
        # produces near-identical consecutive frames (see docs/DECISIONS.md).
        gate_ok = not (hres.black or hres.obstructed)

        mask: Optional[np.ndarray] = None
        if gate_ok:
            mask = self._roi_masked(self.backend.predict_mask(frame_rgb))
            est = estimate_from_mask(mask, self.cfg, self.estimator)
            if not est.valid and est.reason == "no_water":
                # Healthy frame, no water at the pole: a genuine 0 cm reading, not a gap.
                est = DepthEstimate(
                    depth_cm=0.0, waterline_row=est.waterline_row, confidence=DRY_CONFIDENCE,
                    valid=True, calibration_mode=est.calibration_mode,
                    cm_per_pixel_at_waterline=est.cm_per_pixel_at_waterline,
                    uncertainty_cm=est.uncertainty_cm, reason="dry",
                )
        else:
            est = DepthEstimate(
                depth_cm=0.0, waterline_row=0.0, confidence=0.0, valid=False,
                calibration_mode=self.cfg.calibration.mode,
                cm_per_pixel_at_waterline=0.0, uncertainty_cm=0.0,
            )

        fo = self.filter.update(timestamp_s, est)
        so = self.machine.update(timestamp_s, fo)
        depth = float(so.filtered_depth_cm or 0.0)

        annotated = annotate_frame(
            frame_rgb, self.cfg, mask,
            est.waterline_row if est.valid and est.reason == "ok" else None,
            depth if so.filtered_depth_cm is not None else None,
            so.state.value,
            severity=so.severity_class.name,
        )
        frame = LiveFrame(
            index=self._n,
            timestamp_s=timestamp_s,
            raw_depth_cm=float(est.depth_cm) if est.valid else None,
            depth_cm=depth,
            confidence=float(est.confidence),
            valid=bool(est.valid),
            reason=est.reason if gate_ok else "unhealthy_frame",
            state=so.state.value,
            severity=so.severity_class.name,
            rate_cm_per_10min=so.rate_cm_per_10min,
            transition=so.transition,
            health_ok=hres.healthy,
            health_detail=hres.detail,
            annotated_rgb=annotated,
            frozen_flag=hres.frozen,
            black_flag=hres.black,
            obstructed_flag=hres.obstructed,
        )
        self._n += 1
        return frame


def analyze_video(
    video_path: str | Path,
    settings: LiveSettings,
    backend: Optional["SegmentationBackend"] = None,
    store: Optional["EventStore"] = None,
    run_id: Optional[str] = None,
    camera_id: str = "upload",
    synthetic: bool = False,
    snapshot_dir: Optional[Path] = None,
    snapshot_every_s: float = 5.0,
    max_frames: Optional[int] = None,
) -> Iterator[LiveFrame]:
    """Generator: yields one LiveFrame per sampled frame, in video-time order.

    If a store is given, events/alerts/health rows are written as frames are produced
    so the recorded-run dashboard can replay the same analysis later.
    """
    import cv2

    probe_video(video_path)  # fail fast on unreadable files
    first_frame_shape: Optional[tuple[int, ...]] = None
    pipeline: Optional[LivePipeline] = None
    cfg: Optional[CameraConfig] = None
    run_id = run_id or f"upload_{uuid.uuid4().hex[:8]}"
    last_snap = -1e9

    for ts, frame in iter_sampled_frames(
        video_path, settings.sample_fps, settings.max_width, max_frames
    ):
        if pipeline is None:
            first_frame_shape = frame.shape
            cfg = build_camera_config(settings, first_frame_shape, camera_id)
            reference = None
            if settings.mask_source == "change" or settings.stabilise:
                reference = read_reference(video_path, settings.ref_time_s, 5, settings.max_width)
                if reference.shape != frame.shape:
                    reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
            pipeline = LivePipeline(
                cfg, build_experiment_config(settings),
                build_backend(settings, cfg, reference, backend),
                estimator=settings.estimator, reference_rgb=reference,
                stabilise=settings.stabilise,
                stab_rows=(0, int(settings.mark_row)),
            )
            if store is not None:
                store.upsert_run(
                    run_id=run_id, source=str(video_path), camera_id=camera_id,
                    config_hash=cfg.config_hash(), git_commit="n/a",
                    started_at=time.time(), synthetic=synthetic,
                )
            if snapshot_dir is not None:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
        elif frame.shape != first_frame_shape:
            frame = cv2.resize(frame, (first_frame_shape[1], first_frame_shape[0]))

        out = pipeline.step(ts, frame)

        if store is not None and cfg is not None:
            snap_path: Optional[str] = None
            if snapshot_dir is not None and (
                ts - last_snap >= snapshot_every_s or out.transition is not None
            ):
                p = snapshot_dir / f"snap_{ts:.1f}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(out.annotated_rgb, cv2.COLOR_RGB2BGR))
                snap_path = str(p)
                last_snap = ts
            store.insert_health(
                run_id=run_id, camera_id=camera_id, timestamp=ts,
                frozen=out.frozen_flag, black=out.black_flag, obstructed=out.obstructed_flag,
                offline=False, detail=out.health_detail or None,
            )
            store.insert_event(
                run_id=run_id, camera_id=camera_id, timestamp=ts, state=out.state,
                depth_cm=out.depth_cm, severity_class=out.severity,
                confidence=out.confidence, rate_cm_per_10min=out.rate_cm_per_10min,
                snapshot_path=snap_path, model_version=pipeline.model_version,
                config_hash=cfg.config_hash(), synthetic=synthetic,
                valid=out.valid, stale=False,
            )
            if out.transition is not None:
                store.insert_alert(
                    run_id=run_id, camera_id=camera_id, timestamp=ts,
                    transition=out.transition, severity_class=out.severity,
                    depth_cm=out.depth_cm,
                )
        yield out

    if store is not None:
        store.finalize_run(run_id, time.time())
