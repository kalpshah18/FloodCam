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

## 2026-10-21: frozen_k_frames raised from 3 to 5 (A1 regression fix)

**Decision:** `HealthConfig.frozen_k_frames` default is 5 (requires 4 consecutive sub-threshold
inter-frame diffs before declaring a frame frozen), up from the original 3 (2 consecutive diffs).

**Alternative:** Lower `frozen_diff_threshold` from 5.0 to ~3.0 so JPEG-compressed slightly-varying
frames no longer cross the threshold.

**Reason:** JPEG save/load (quality=95) compresses subtle per-frame RNG differences in the
synthetic renderer.  For a 900-frame slow-rise scene, occasional consecutive frame pairs have
mean |diff| of 4.34–4.59 — just below the 5.0 threshold — producing 2–3 false "frozen"
detections per run.  These are isolated one-off pairs (max count=2), not sustained freezes.
A truly frozen camera produces diff=0.00 for many consecutive frames (count >> 10 in tests).
Raising k to 5 requires 4 consecutive sub-threshold pairs before flagging.  JPEG artifacts
are sporadic and never reach count=4; genuine frozen cameras stay frozen for many minutes.
Lowering the threshold to 3.0 would also work but risks missing near-frozen real cameras
with slight compression noise.  The k increase is more conservative and domain-appropriate.
Verified: health_fault frozen segment (diff=0.00 for 10 frames) still flagged at t=75 (k=5)
vs t=73 (k=3) — 2 frames later, still within the same fault window.

---

## 2026-10-21: Alert deduplication by transition string

**Decision:** A new alert row is written only when the transition string changes from the previous
alert in a run.

**Alternative:** Alert on every state transition regardless of deduplication.

**Reason:** Prevents flooding the alert table if the state machine rapidly cycles through the
same pair of states (e.g., Normal→Pending→Normal cycles near the threshold). Each unique
transition pattern fires once per occurrence in the natural flow.

---

## 2026-10-21: M8 - SegFormer model, loss, and stack

**Decision:** SegFormer-B0 default (B1 via config), 2 output classes (background, water),
encoder initialised from ImageNet (`nvidia/mit-b0`) with a random decode head, or from an
ADE-pretrained checkpoint if `init_weights` says so. The choice is recorded in `meta.json`.

**Alternative:** U-Net with a pretrained encoder.

**Reason:** CLAUDE.md names SegFormer first; it is small enough for a T4, and the HF
implementation needs no extra dependency. Loss = class-weighted cross-entropy + soft Dice:
water is a minority class in most scenes (and ~0 in negatives), so CE alone drifts toward
"all background"; Dice optimises the overlap that IoU/F1 measure. Class weights
(`N / (2 * count_c)`) are computed from the training split only (a guard raises if a non-train
row is passed). ImageNet mean/std are constants of the pretrained encoder, not fitted to our
data, so there is no dataset-normalisation statistic to leak. Validation IoU/F1 are
accumulated over the whole split (not a mean of per-image scores) at the training resolution
(`image_size`, default 512, square resize), so they are not full-resolution numbers.

## 2026-10-21: M8 - Class mapping to binary water

**Decision:** The source-class -> water mapping lives in `configs/seg_classmap.yaml`. Class
names are NOT trusted from memory: the prepare script parses each dataset's own label file and
aborts (`ClassMapError`) if a mapped name or id is absent. The names in the YAML are
candidates until the first `--verify` against real files. Layout of each dataset under
`data/public/` is the layout this project expects (user moves/symlinks the download), not a
claim about the official archive structure.

**Alternative:** hardcode ids (e.g. ADE "water" = N).

**Reason:** a wrong id silently trains on the wrong class; a loud failure costs one minute.
Mapping all water-like classes (sea, river, lake, ...) to one "water" class matches the task
("is this pixel standing/flowing water"); rivers/canals are excluded from the *flood* decision
later by ROI and static-region masks, not by the segmenter. Official train/val splits are used
as-is for public sets; negatives (no official split) are split by hashing the sequence folder
so near-duplicate frames never straddle train/val.

## 2026-10-21: M8 - Domain gap: FloodNet off, negatives on, augmentation

**Decision:** FloodNet is disabled by default (`enabled: false` + `domain_gap_note`).
Hard negatives (puddles, wet roads, reflections, rain) are supported with empty or partial
masks and a sampling weight (default 2.0). Augmentations: horizontal flip, scale/crop, colour
jitter, blur, noise, JPEG, rain streaks; no vertical flip. Fine-tuning on self-labeled
ground-level frames is the stage that closes the gap to our camera.

**Alternative:** pretrain on everything including FloodNet.

**Reason:** FloodNet is aerial UAV imagery; viewpoint, scale and water appearance differ from a
fixed ground-level camera, so it can pull the encoder toward the wrong cues. It remains
available behind a flag for an explicit ablation. Vertical flips would invert the gravity prior
(water below the waterline) that the geometry relies on.

## 2026-10-21: M8 - Training guards, ROI, and runner coupling

**Decision:** (1) finetune refuses synthetic rows unless `--smoke-synthetic`, and refuses any
scene named `locked_test_scene` in any `data/splits/*.json` (or `locked_scenes` in the config);
pretrain refuses synthetic rows outside `--smoke`. (2) `SegFormerBackend` does NOT apply the ROI;
the caller does, as specified. (3) `estimate_from_mask()` was split out of `estimate()` so any
backend's mask can feed the same waterline/calibration code (HSV path unchanged).

**Known limitation (not fixed this milestone):** `floodcam.run.process_timelapse` calls
`estimate()` (HSV inside) for depth and uses the backend mask only for the snapshot overlay and
`model_version`. So the runner does not yet produce SegFormer-based depth; `scripts/eval_seg.py
--with-depth` does (via `estimate_from_mask`). Wiring the runner is a small follow-up that was
outside this milestone's scope.

**Licences:** `license_verified` is false for every dataset until a human flips it; the
licence strings previously listed in `AI_USE.md` were written from memory and are unverified.

## 2026-10-25: Upload-video live dashboard (`app/upload_view.py`, `floodcam.live`)

**Decision:** The dashboard's default mode takes an uploaded video, lets the user calibrate on a
frame, and updates depth/severity/state per sampled frame while the video is analysed. Replay of
recorded runs remains as a second mode. Calibration for an arbitrary video is a 2-point
`LinearCalib` (ground-contact row = 0 cm, one marked row at a known height), entered with
sliders; the preview shows a test reading so the user can check it on a frame with known depth.
**Alternative:** click-to-mark with a custom component (extra dependency, "ask before adding"),
or homography (needs 4+ points; still available through `floodcam.calib.annotate`).

**Decision:** Timestamps are video time (`frame_index / fps`), not wall-clock. Playback speed only
paces the display. Persistence / trend windows default to seconds (10 s / 20 s) in this mode so
a short clip can reach Flood and Recovering; the sidebar says to use minutes for real events.
Rate-of-rise severity escalation is off by default here: over a 20 s window depth noise
converts to a large cm/10 min rate. The Flood/Recovering logic remains a demonstration, not a
validated claim (CLAUDE.md).

**Decision:** A healthy frame with no water at the pole is a valid 0 cm reading
(`reason="no_water"` -> depth 0, confidence 0.8), not an excluded sample. Without this the
temporal filter held the last median after water receded (observed: 2.3 cm shown on a dry scene).
`DepthEstimate` gained a defaulted `reason` field to carry the waterline reason.

**Decision:** In this pipeline only black/obstructed frames gate the depth estimate. The frozen
check (mean abs diff < threshold for k frames) is reported but does not gate: a static
fixed-camera scene legitimately yields near-identical frames, and the default threshold (5.0)
would flag real static footage. Uploads use threshold 0.5.

**Limitation:** The waterline detector reads water on the columns flanking the pole and requires
water at the ground-contact row; it is the project's HSV baseline geometry and has only been
validated on synthetic video. No real-footage accuracy claim exists.

## 2026-10-26: Real-footage robustness for upload mode

**Decision:** Default depth reading is the "gauge" method: a narrow strip on the reference
object itself, thresholded for change against a dry reference frame (mean of 5 frames; Lab dE
after per-frame exposure compensation from rows above the highest mark). The waterline is the
top of the contiguous changed run above the ground row (gaps up to 1 cm bridged).
**Alternative:** global HSV colour threshold + reading water beside the pole. Kept as an option
(and for the synthetic legacy tests) but a colour threshold cannot separate water from sky,
asphalt or foliage. Reading beside the pole also does not measure depth on a real ground plane.
**Limits:** needs a dry moment in the video; wet-looking but unchanged water (clear, still,
no reflection) can be missed; shadows, passers-by and reflections cause false change. None of
this is validated on real footage yet.

**Decision:** Small camera shake is removed by phase correlation on edge magnitude, using only rows
above the highest calibration mark (never underwater), accepted only if it improves a z-scored
alignment error by >= 15%. Using the whole frame was tried first and failed: rising water was read as
camera motion (up to 9 px drift).

**Decision:** Click-to-mark uses `streamlit-image-coordinates` (small pure-Python component).
Sliders remain as fine-tune and as the fallback if the package is missing.

**Decision:** Annotated video is encoded with ffmpeg (H.264) when available, else mp4v.

## 2026-10-28: Real USGS footage pipeline (supersedes synthetic-only evaluation)

**Decision:** Evaluate on real fixed-camera footage with official gauge labels (USGS cameras, see
`docs/DATA_SOURCES.md`). Synthetic data stays for unit tests and the upload-dashboard demo only.
**Why:** no self-collected data exists; the USGS S3 bucket was the only reachable real source.

**Decision:** Read depth from the staff plate with a learned per-row classifier (gradient boosting
on colour-change, texture and context features of the rectified plate strip, labels derived from the
gauge) and a monotone change-point decode, instead of the vertical-strip HSV/change gauge. **Alternative
kept as baseline:** the same decode on a thresholded dry-template colour change (threshold tuned on
training blocks). **Why:** the plate is tilted and its appearance changes by season, lighting and
infrared; a fixed threshold fails (see results/real/SUMMARY.md). No GPU or torch is needed.

**Decision:** Split by calendar-month block (an event/session). Locked test blocks 2025-04 and 2026-04
were fixed in `scripts/make_usgs_splits.py` before any model was trained. Tuning (HSV-free: baseline
threshold, gate, registration references) used dev blocks only. Registration reference frames are
chosen from dev blocks using image geometry only.

**Decision:** Registration = SIFT homography to the best of 13 reference views (one per dev month), with
a causal fallback that reuses the homography of the most recent registered frame (camera pose changes
rarely). **Limit:** a pose change during a run of failed frames is not detected.

**Decision (design change after seeing dev results, disclosed):** the image-quality gate was first
one-sided (Laplacian variance >= 0.5 x median, to catch blur). Dev-block error vs Laplacian
variance showed lens dirt/spray frames with variance 3000+ also fail, so the gate became two-sided
[0.55, 1.7] x training median. The locked blocks had already been evaluated once with the one-sided
gate; the run was repeated with `--force` and both results are kept
(`results/real/final_v1_gate_one_sided_metrics.json`, `results/real/final_test_log.jsonl`). The gate
is chosen from dev data only, but the second locked run is therefore not a pristine single look.

**Decision:** Resolution sweep isolates the readout resolution: registration is computed once at native
resolution, then the frame is area-downsampled by f and the plate re-sampled at 1/f rows. Registering at the
degraded resolution was tried first and collapses at f >= 8 (registered fraction 34% at f=8, 6% at f=16,
0% at f=32), which mixes a calibration failure with the question asked. Treat the sweep as "how coarse
can the plate readout be once the camera is calibrated".

**Limits to state in the report:** one camera/site (n = number of month blocks, 13 dev + 2 locked);
the gauge sits beside the plate; depth below the plate bottom (~4 cm) is invisible, so values < ~4 cm
are reported as 0; infrared and night frames share the model but are a different appearance; the
labels (gage height) are not tape measures.
