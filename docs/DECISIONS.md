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

## 2026-10-21: Half-plane water mask in synthetic renderer is a known limitation

**Decision:** Document and accept that the synthetic renderer's water mask is a full-width horizontal half-plane.

**Alternative:** Render a more realistic mask with occlusion, curved surface, and texture variation.

**Reason:** The synthetic mask is intentionally simple to make unit tests deterministic and geometry-based. It is sufficient for calibration tests and pipeline integration tests. However, segmentation metrics (IoU, F1) on synthetic masks are trivially ~100% because there is no boundary noise. This is documented in CALIBRATION.md and must not be reported as a result — only real-data masks count for segmentation evaluation.

---
