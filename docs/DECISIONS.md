# Design Decisions

Each entry: decision made, alternative considered, reason.

---

## 2026-10-21: Config system uses dataclasses + PyYAML instead of Pydantic

**Decision:** Python stdlib dataclasses + PyYAML for configuration.

**Alternative:** Pydantic v2 for automatic validation and JSON schema generation.

**Reason:** Pydantic is not in the listed stack. Dataclasses with explicit `from_dict` methods provide sufficient type safety for this project's config complexity without an extra dependency. Can be migrated to Pydantic later if needed without changing the YAML format.

---

## 2026-10-21: Split by scene, not by frame

**Decision:** All train/val/test splits are partitioned at the scene/session level, never at the frame level.

**Alternative:** Random frame-level 80/10/10 split.

**Reason:** Consecutive frames from the same scene are near-duplicates (temporal correlation). Frame-level splitting leaks near-identical frames across partitions, inflating all reported metrics. Scene-level splits give a fair estimate of generalization to unseen lighting conditions and settings.

---

## 2026-10-21: Synthetic data used for tests and demos only

**Decision:** The synthetic generator (`src/floodcam/synth/`) produces artifacts tagged `synthetic=true` in manifests. All evaluation scripts exclude synthetic scenes.

**Alternative:** Include synthetic data to augment training and report results on it.

**Reason:** Reporting metrics on synthetic data would not answer the research question (real shallow-water depth estimation). Synthetic data is valuable for CI and unit tests but must never appear in results tables. The tag ensures the split generator and eval scripts can enforce exclusion automatically.

---

## 2026-10-21: Hysteresis band: trigger 3.5 cm, clear 2.5 cm, persistence 5 min

**Decision:** Use a 1 cm hysteresis band around the 3 cm flood threshold.

**Alternative:** Hard threshold at exactly 3 cm.

**Reason:** Waterline localization error of ~1-2 px at typical resolutions corresponds to ~0.5-1.5 cm depth error. A dead band prevents rapid state oscillation near threshold. The 5-minute persistence requirement further filters transient detections (splash, reflections).

---

## 2026-10-21: Locked test scene evaluated exactly once via scripts/final_test.py

**Decision:** One whole scene is locked as the final test set and evaluated only once through `scripts/final_test.py`, which refuses to re-run without `--force` and logs every run.

**Alternative:** Use k-fold cross-validation and report average across all folds.

**Reason:** Repeated evaluation on the test set leads to implicit over-fitting through hyperparameter choices. Locking one scene and running evaluation once preserves the integrity of the final reported number. Leave-one-scene-out CV is used for all tuning decisions.

---

## 2026-10-21: Pinhole camera model for synthetic renderer

**Decision:** Use a simple pinhole camera model (no lens distortion) for the synthetic renderer.

**Alternative:** Full OpenCV distortion model.

**Reason:** The research question concerns geometric depth accuracy, not lens distortion. The analytic formula `dv/dh = -f*D/Z_c(h)^2` (derived cleanly from the pinhole model) is the exact ground truth for calibration and depth unit tests. Adding lens distortion would complicate the analytic waterline check without improving experiment validity.

---

## 2026-10-21: Camera-frame convention Y_c = (H-h)*cos(alpha) - D*sin(alpha)

**Decision:** Y_c is positive downward (standard camera convention); depression angle alpha tilts the optical axis below horizontal.

**Reason:** This makes `dv/dh = -f*D/Z_c^2` always negative (waterline moves up in image as depth rises), matching physical intuition. The derivation simplifies to a closed-form independent of h because cross terms cancel exactly.

---

## 2026-10-21: Synthetic splits use a separate file (synthetic_splits.json)

**Decision:** LOSO folds for synthetic data go in `data/splits/synthetic_splits.json`, never in `train.json`/`val.json`/`test.json`.

**Alternative:** Reuse the standard split files with a type tag.

**Reason:** Real-data splits must be immutable once generated. Mixing synthetic LOSO folds into the same file would force regeneration of real splits whenever synthetic parameters change. Separate files keep the two regimes independent and prevent accidental overwrite.

---

## 2026-10-21: sigma_s definition for LinearCalib

**Decision:** `sigma_s` is the OLS standard error of the slope `a` (not sigma of the scale as a separate fit):
`MSE = sum(residuals²) / (n−2)` for n > 2; `sigma_s = sqrt(MSE / sum((v_i − v_mean)²))`.

**Alternative:** Bootstrap sigma_s from 100 resamples; or define as half the calib residual range.

**Reason:** OLS std error is exact for Gaussian residuals, produces a closed-form formula with well-understood propagation into the depth uncertainty equation, and costs nothing extra. Bootstrap is more general but expensive and unnecessary for the small (~5-point) calibration sets used here.

---

## 2026-10-21: Linear mode for tilt < 15°, homography for tilt ≥ 15°

**Decision:** Linear calibration (`h = a·v + b`) is recommended for camera tilt < ~15°. Homography with a 2D calibration board is required for higher tilts.

**Alternative:** Always use homography; linear is never recommended.

**Reason:** For a 20° tilt scene with calib marks at [0,10,20,30,40] cm, the linear model has ~0.3 cm noiseless error at depths 2–3 cm (the flood threshold zone). Homography achieves < 0.01 cm. Linear is simpler to annotate (pole-only marks), but users must understand it is approximate. The annotation CLI enforces ≥ 2 points (linear) and ≥ 4 non-collinear 2D points (homography).

---

## 2026-10-21: Calibration fit has no access to evaluation labels (by construction)

**Decision:** `LinearCalib.fit()` and `HomographyCalib.fit()` accept only explicit `(pixel_row, height_cm)` or `(u, v, x_cm, h_cm)` pairs. No frame data, manifest rows, or depth labels are passed in.

**Alternative:** Pass a manifest and let fit() filter calibration-tagged frames internally.

**Reason:** Data-flow isolation prevents accidental leakage of evaluation-frame depth labels into calibration. The tight API signature makes this obvious and testable. The annotation CLI is the only place that reads frame data (for interactive display only, not for fitting).

---

## 2026-10-21: Waterline confidence formula

**Decision:** `confidence = continuity_frac × coverage_frac × resolution_factor`
where `resolution_factor = min(1.0, 0.5 / cm_per_pixel)`.

`continuity_frac` is the inter-column consistency of waterline row estimates:
`max(0, 1 - std(per_col_waterline_rows) / 20 px)`. A standard deviation greater
than 10 px (continuity_frac < 0.5) triggers `reason="broken_mask"`.

**Alternative:** Single threshold on per-column water-region fill fraction.

**Reason:** The fill-fraction approach (water in [r, r_g] / span) always yields 1.0
because the upward contiguous scan by construction finds a fully-water segment.
Inter-column consistency instead detects the physically important failure mode:
different columns seeing wildly different waterline heights (reflections, partial
occlusion, large mask holes), while clean synthetic masks give std ≈ 0 → 1.0.
The `resolution_factor = min(1.0, 0.5 / s)` penalises coarse resolution where
3 cm detection is unreliable (reference: 6 px per 3 cm at the minimum usable
resolution of 0.5 cm/px).

---

## 2026-10-21: Flood state machine transition table

**Decision:** Four states (Normal, Pending, Flood, Recovering) with timestamp-based hysteresis and persistence.

Key choices:
- Flood can only reach Normal through Recovering (prevents spurious cleared alarms).
- Falling trend while Pending resets to Normal, not Recovering — since Flood was never declared.
- Rate=None (insufficient data) does not trigger Flood→Recovering; we wait for data.
- Draining logic (Recovering state) is a dashboard demonstration only, not a validated claim, because test clips are too short to validate minutes-long trends (see CLAUDE.md).

**Alternative:** Single-threshold hysteresis without a Recovering state.

**Reason:** Recovering lets the dashboard distinguish "water is draining" from "alarm cleared". The Flood→Normal-only-through-Recovering rule is critical for alert integrity — without it, a momentary low reading (reflection, shadow) could clear an active flood alarm.

---

## 2026-10-21: Half-plane water mask in synthetic renderer is a known limitation

**Decision:** Document and accept that the synthetic renderer's water mask is a full-width horizontal half-plane.

**Alternative:** Render a more realistic mask with occlusion, curved surface, and texture variation.

**Reason:** The synthetic mask is intentionally simple to make unit tests deterministic and geometry-based. It is sufficient for calibration tests and pipeline integration tests. However, segmentation metrics (IoU, F1) on synthetic masks are trivially ~100% because there is no boundary noise. This is documented in CALIBRATION.md and must not be reported as a result — only real-data masks count for segmentation evaluation.

---

## 2026-10-21: Oscillation and hysteresis behavior

**Finding:** `oscillation_around_threshold(center=3.0, amplitude=1.5)` swings depth [1.5, 4.5] cm.
Since 4.5 > trigger_cm (3.5) and 1.5 < clear_cm (2.5), Normal↔Pending transitions are **correct
behavior** — not chatter. The test verifies alert-safety properties: Flood never reached (60 s
oscillation period << 5 min persistence) and no single Pending dwell approaches persistence.

An earlier test used `period_s=200` to cap transition count at ≤8; this was a workaround that
obscured correct behavior. The test was rewritten to use the default 60 s period and assert
the meaningful properties only.

---

## 2026-10-21: Pending timer across a stale gap — wall-time counting

**Decision:** `pending_since` counts wall time including dropout gaps. A gap does not reset the timer.

**Alternative:** Reset `pending_since` on gap (conservative; timer restarts after dropout).

**Reason:** Two safeguards make wall-time counting safe:
(1) `stale=True` during and immediately after the gap → no Pending→Flood transition can fire;
(2) `rate=None` after a gap (insufficient history) → the `rate > falling_slope` check fails,
blocking the transition until enough data rebuilds.
Resetting the timer would unfairly delay a legitimate Flood declaration after a brief camera
dropout during a real flood event, which is worse than the alternative.

---

## 2026-10-21: Runner uses timelapse-dir source only (video deferred)

**Decision:** Milestone 6 runner reads from a timelapse directory (timestamps.json + frames/*.jpg).
Video (RTSP or file) is deferred to a later milestone.

**Alternative:** Implement video decoding immediately (cv2.VideoCapture).

**Reason:** Timelapse dirs are deterministic, testable without a video file, and sufficient for
all current evaluation needs. Adding video decoding now would introduce timestamp extraction
complexity without benefiting the current research question.

---

## 2026-10-21: Synthetic flag must be set explicitly; runner refuses to mix

**Decision:** The `synthetic` column on every event row must match the run's declared flag.
Timelapse sources auto-set synthetic=True (via timestamps.json). Any other source defaults to
False. Mixing synthetic and real events in one run is refused.

**Reason:** Accidental mixing would contaminate real-data runs with synthetic results.

---

## 2026-10-21: Alert deduplication by transition string

**Decision:** A new alert row is written only when the transition string changes from the previous
alert in a run.

**Alternative:** Alert on every state transition regardless of deduplication.

**Reason:** Prevents flooding the alert table if the state machine rapidly cycles through the
same pair of states (e.g., Normal→Pending→Normal cycles near the threshold). Each unique
transition pattern fires once per occurrence in the natural flow.

---
