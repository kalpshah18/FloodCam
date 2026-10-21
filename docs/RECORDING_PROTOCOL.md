# Recording Protocol

> This document must be completed before milestone 10 (real-data hooks).
> Current content is a planning stub.

## Purpose

Guide for recording the self-collected depth dataset used for final evaluation.
Follow these steps exactly to ensure frames are usable for calibration and depth estimation.

---

## Equipment checklist

- [ ] Fixed camera (phone or webcam) on a stable tripod or clamp — no movement during a session.
- [ ] Rigid reference pole or vertical marking with known height increments (every 1 cm or 5 cm).
  Recommended: a white PVC pipe with black tape marks at 0, 2, 3, 5, 10, 20 cm from the base.
- [ ] Shallow tray or controlled water patch (at least 30 × 30 cm, dark or neutral floor).
- [ ] Tape measure.
- [ ] Consistent lighting source (avoid direct sun reflections; diffuse indoor light preferred).

---

## Camera mounting

1. Mount the camera at a fixed height (40–80 cm from ground) and angle (15–45° depression).
2. Ensure the full reference pole is visible in the frame with at least 5 cm of ground visible below it.
3. Record the camera position in the session manifest.
4. Do not move the camera during a session.

---

## Session structure

Each **session** = one camera position + one lighting condition.
Minimum: 3 sessions (varied lighting: indoor diffuse, indoor directional, outdoor overcast).

### Within a session

1. Start recording at **depth = 0 cm** (no water). Hold for ≥ 10 seconds.
2. Add water incrementally: 2 cm, 3 cm, 5 cm, 10 cm, 20 cm. Hold each level for ≥ 10 seconds.
3. Measure each level with the tape measure against the pole.
4. For negative conditions: record a separate clip with rain droplets, puddle reflections,
   or direct glare on the reference object.

---

## Naming scheme

```
data/raw/<session_id>/
    frames/        # extracted at 1 fps by scripts/extract_frames.py
    video.mp4      # original recording
    manifest.json  # see schema below
```

Session ID format: `YYYYMMDD_<location_code>_<lighting_code>_<take>`
Example: `20261101_lab_indoor_diffuse_01`

---

## Manifest schema (one per session)

```json
{
  "scene_id": "20261101_lab_indoor_diffuse_01",
  "camera_id": "cam_lab",
  "recorded_date": "2026-11-01",
  "lighting": "indoor_diffuse",
  "background": "gray_floor",
  "synthetic": false,
  "depths_cm": [0, 2, 3, 5, 10, 20],
  "fps": 1,
  "frame_count": null,
  "notes": ""
}
```

---

## What NOT to record

- People, faces, or private locations.
- Moving cameras.
- Submerged cameras.
- Water deeper than ~25 cm (exceeds depth range).

---

## After recording

1. Extract frames: `python scripts/extract_frames.py --session <session_id>`
2. Verify manifest: `python scripts/verify_manifest.py --session <session_id>`
3. Label masks for 10–20 frames per session: `python scripts/label_tool.py --session <session_id>`
4. Add session to `data/manifest.json`.
5. Regenerate splits only if adding new sessions before final evaluation.
