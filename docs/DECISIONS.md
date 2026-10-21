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
