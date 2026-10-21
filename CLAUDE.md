# CLAUDE.md

## Project
ES 678 (graduate ML) course project. Student: Kalp Shah, 24110152.
Title: Estimating Shallow Flood Depth from Fixed-Camera Video via Water Segmentation and Calibrated Geometry.

## Deliverable in this phase
A reproducible ML pipeline plus a dashboard. The LaTeX paper is NOT part of this phase. Do not write or scaffold the paper. The paper will be written later from this repo, so every result, split, config, and decision must be stored in files that a later session can read.

Pipeline: video -> frame sampling -> water segmentation -> ROI -> calibrated depth -> temporal filter -> state machine -> SQLite event log -> Streamlit dashboard.

## Research question (drives all experiments)
Can water depth near a reference object (marked pole, curb, wall) be estimated from a fixed-camera frame accurately enough to decide whether a 3 cm flood threshold is exceeded? How does depth error depend on resolution (cm/pixel) and segmentation quality? At what resolution does 3 cm detection stop being reliable?

## Targets
1. Binary water mask per frame.
2. Depth h in cm (regression).
3. Flood indicator 1[h >= 3 cm], with hysteresis (trigger >= 3.5 cm, clear < 2.5 cm).
Severity classes (configurable per camera): Normal <3, Advisory 3-10, Warning 10-25, Severe 25-50, Critical >50 cm. Escalate one class if rate of rise exceeds 5 cm per 10 min.

Prediction time: per sampled frame (default 1 fps), using the current and past frames plus a one-time camera calibration. Tape-measured depth is never available at prediction time.

## Flood state definition (dashboard logic)
Inside a manually drawn ROI, a flood is declared when estimated depth >= 3 cm, persists for N minutes (default 5, configurable), and the trend is flat or rising. A steadily falling level is "Recovering", not flood. Interpretation of the original brief: "accumulated water that is draining" means accumulated and NOT draining. Puddles (small area, depth < 3 cm), rivers, canals, and drains are excluded through ROI and static-region masks.
State machine: Normal -> Pending -> Flood -> Recovering -> Normal, hysteresis on every transition.

## Claim limits
Fixed camera, visible reference object of known size, manual ROI, controlled recordings. Not real floods, not moving cameras, not arbitrary video. The state machine's draining-vs-accumulating logic is a dashboard demonstration only. It is not an evaluated claim, because short clips cannot validate minutes-long trends. Do not present it as validated.

## Data
- Public segmentation: ADE20K, COCO-Stuff, FloodNet, Kaggle/Roboflow flood sets. Check each license. One row = image + pixel water mask.
- Self-collected depth set (human task): shallow water in a tray or controlled patch beside a marked pole; tape-measured depths 0, 2, 3, 5, 10, 20 cm; >= 3 scenes with varied lighting and background; ~500 frames. One row = frame + tape-measured h. Effective sample size = number of scenes/sessions, not frames.
- Self-labeled masks (human task): 50-200 ground-level frames.
- Negative footage: rain, puddles, reflections, night.
- No people or private locations in recordings. Never commit raw data, credentials, or API keys. Commit only small safe samples or precomputed outputs.
- No real data exists at project start. Build and test everything first on a synthetic generator (`src/floodcam/synth/`: renders a pole, ground plane, and a water layer at known depth). Synthetic data is for unit tests, integration tests, and dashboard demos only. It must never appear in reported results. Tag every synthetic artifact with `synthetic=true` in manifests.

## Methods
- Baseline: HSV color threshold mask + same geometric depth conversion. Trivial flood reference: always "no flood".
- Candidate 1: SegFormer-B0/B1 (or U-Net, pretrained encoder), trained on public data, fine-tuned on self-labeled ground-level frames, then calibrated geometry: h = s (v_g - v_w), v_w = waterline row on reference object, v_g = ground-contact row, s = cm/pixel fitted from marked reference points. Use a homography if camera tilt is significant.
- Candidate 2: small CNN (pretrained backbone) classifying depth class from a crop around the reference object.
- Every estimate carries a confidence score (mask quality at the pole intersection, occlusion, local resolution). Low-confidence frames are logged and excluded from the temporal filter.

## Evaluation rules (graded later; do not violate)
- Define splits before comparing models. Split by scene/session, never by frame (consecutive frames are near-duplicates).
- Splits live in `data/splits/*.json`, generated once by a seeded script and committed (IDs only, no images).
- Segmentation: official train/val splits for public data; final test on self-labeled frames from scenes excluded from fine-tuning.
- Depth: leave-one-scene-out CV for tuning (hysteresis thresholds, HSV params, epochs). Lock one whole scene/session as the final test set; evaluate on it once, through a single script (`scripts/final_test.py`) that refuses to run twice without an explicit `--force` and logs every run.
- Fit calibration on calibration frames only. Fit normalization and preprocessing statistics on training data only.
- Metrics: IoU and F1 (masks); MAE per depth bin, with error near 3 cm reported separately; precision, recall, false alarms per hour on negative footage. Plain accuracy is never a headline metric.
- Uncertainty = spread across held-out scenes, not pooled-frame intervals.
- Compare baseline vs segmentation model vs classifier on the same held-out scenes.

## Depth-analysis experiments (implement as scripts; paper uses their outputs)
Derive depth error from pixel quantization and calibration error:
sigma_h^2 ~ s^2 sigma_v^2 + (v_g - v_w)^2 sigma_s^2 + sigma_g^2
(sigma_v = waterline localization error in px, sigma_s = scale-fit error, sigma_g = ground-reference error). Validate empirically: (a) downsample frames to vary s, (b) erode/dilate predicted masks to vary sigma_v, (c) ablate SegFormer mask vs HSV baseline. Outputs saved to `results/`: max cm/pixel at which 3 cm is reliably detectable; derived vs measured error; failure-mode audit of the worst frames (reflections, glare, occlusion, night) with frame IDs and saved overlays.

## Dashboard (Streamlit, reads SQLite; no scaling concerns)
Panels: live status per camera with severity color; depth time-series with threshold lines; latest annotated frame (mask overlay + waterline + reference object); alert log; camera health (frozen frame, black frame, lens obstruction, offline). Sidebar: choose camera, select a recorded run, replay speed. Must work in replay mode on recorded video files and on synthetic demo runs. Live RTSP is optional and last.
Alerts: console and webhook stub only.

Event record (one SQLite row per processed frame): camera_id, timestamp, state, depth_cm, severity_class, confidence, rate_cm_per_10min, snapshot_path, model_version, config_hash, synthetic flag.

## Stack
Python 3.11, PyTorch, Hugging Face transformers (SegFormer), OpenCV, NumPy, scikit-learn, Streamlit, SQLite (stdlib), pytest, ruff. Pin versions in `requirements.txt`. YAML configs. Ask before adding any other heavy dependency.

## Repo layout
```
configs/            per-camera and experiment YAML
data/               gitignored except splits/ and README.md (download and recording instructions)
docs/               RECORDING_PROTOCOL.md, CALIBRATION.md, EXPERIMENT_LOG.md, DECISIONS.md
src/floodcam/       synth/, io/, seg/, calib/, depth/, temporal/, store/, eval/
app/                dashboard.py
scripts/            one script per experiment; each writes to results/
results/            metrics JSON/CSV, figures, overlays (small files only), one subfolder per run
tests/              calibration math, split leakage, state machine, hysteresis, store round-trip
AI_USE.md
Makefile            targets: setup, test, synth-demo, train-seg, eval, dashboard
```
CLI entry: `python -m floodcam.run --source <video> --camera <id> --config <yaml>` writes events and snapshots; `streamlit run app/dashboard.py` shows them.

## Milestones (each ends with passing tests and a commit)
1. Scaffold: repo layout, requirements, Makefile, config loader, seeding utility, CI-style `make test`.
2. Synthetic generator + scene manifest schema + split generator + split-leakage test.
3. Calibration module (linear and homography) with unit tests on synthetic ground truth; CLI helper to click reference points and draw ROI, saving to the camera YAML.
4. HSV baseline mask + waterline detection + depth conversion; end-to-end on synthetic.
5. Temporal filter (rolling median, rate of rise) + state machine + tests for hysteresis, persistence, Recovering.
6. SQLite store + `floodcam.run` + health checks.
7. Dashboard with replay on synthetic runs.
8. SegFormer training script (public data, then fine-tune) with saved metrics and checkpoints outside git.
9. Classifier fallback.
10. Real-data hooks: manifest ingestion for the self-collected set, leave-one-scene-out runner, `final_test.py` lock. Pause here until real data exists.
11. Depth-analysis experiments and failure audit scripts.
12. `scripts/make_figures.py` that regenerates every figure and table from `results/`.

## Human tasks Claude Code cannot do
Recording the ruler-in-water video, tape-measuring depths, labeling masks, choosing calibration points on real footage. Write `docs/RECORDING_PROTOCOL.md` with exact steps (camera mounting, pole marking, depths, lighting variations, naming scheme, manifest fields) before milestone 10. Stop and report when blocked on these.

## Working rules
- No leakage: tests assert no scene appears in more than one split, and that preprocessing statistics are computed from the training split only.
- Every reported number comes from a script and a saved results file. Never hand-type metrics.
- Fix seeds; log config, git commit, and data manifest hash per run in `results/<run>/meta.json`.
- Do not tune on the locked test scene. Flag any code path that touches it.
- Keep `docs/DECISIONS.md` current: each non-obvious choice, the alternative, and the reason. Keep `docs/EXPERIMENT_LOG.md`: date, run ID, what changed, outcome. A later session writes the paper from these.
- Keep `AI_USE.md` listing tools and kinds of assistance. Cite external code and model weights with license.
- Ask before downloading large datasets or installing heavy dependencies.
- Commit small, one milestone step at a time. Never commit data, checkpoints, `.env`, or keys.
