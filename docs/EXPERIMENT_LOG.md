# Experiment Log

Format: Date | Run ID | Change | Outcome

---

| Date | Run ID | Change | Outcome |
|------|--------|--------|---------|
| 2026-10-21 | scaffold-001 | Initial repo scaffold: layout, requirements, Makefile, config system (dataclasses + YAML), seeding utility, split-leakage test. | Milestone 1 complete. All tests pass (`test_splits`, `test_config`, `test_seed`). |
| 2026-10-21 | synth-001 | Milestone 2: pinhole synthetic renderer, scene manifest schema (JSONL), synthetic dataset generation (8 normal + 2 negative scenes, 106 frames), LOSO split generator. | 57/57 tests pass. Locked test: `synth_scene_01`, 7 LOSO folds. |
| 2026-10-21 | calib-001 | Milestone 3: LinearCalib (OLS h=a·v+b), HomographyCalib (normalized DLT), depth_uncertainty_cm (σ_h formula), annotation CLI (headless + interactive). Added intercept_cm and sigma_s to CalibrationConfig; added CameraConfig.save(). | 100/100 tests pass. Linear noiseless error < 0.5 cm (20° scene); homography < 0.01 cm. Monte Carlo 1000-trial noise validation within ±10%. Note: synth_scene_01 was used only for geometric unit tests (project_height calls), not for tolerance choices. |
