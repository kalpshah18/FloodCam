# Camera Calibration Guide

## Overview

FloodCam estimates flood depth from the position of the waterline on a reference
object (marked pole, curb stripe, or wall). The depth formula is:

```
h = s · (v_g − v_w)
```

where:
- `v_g` — ground-contact row of the reference object (pixels from top of frame)
- `v_w` — waterline row at the reference object (pixels from top)
- `s` — scale factor in cm/pixel, fitted from reference marks of known height

---

## Calibration modes

### `linear` (recommended for tilt < ~15°)

Fits `h = a·v + b` from ≥ 2 (pixel_row, height_cm) pairs by OLS
(ordinary least squares). The scale is `s = |a|` (constant across the frame).

**Accuracy:** systematic error from the perspective projection grows with camera
tilt and depth range. For the default 20° tilt scene with calib marks at
0, 10, 20, 30, 40 cm, the maximum noiseless depth error is ~0.3 cm at depths
2–3 cm. Using dense marks that include the target depth range reduces this.

### `homography` (for tilt ≥ ~15° or when higher accuracy is required)

Fits a full projective homography from ≥ 4 non-collinear (image_u, image_v,
plane_x_cm, height_cm) correspondences using normalized DLT. The scale
`s = |dy_cm/dv|` is computed locally at the query pixel. Models perspective
exactly: noiseless depth error is < 0.01 cm on synthetic data.

**Required setup:** calibration marks must span a 2D area on the reference
plane (different horizontal offsets as well as different heights). A single
column of marks at x=0 (pole centre) is collinear and will be rejected.

---

## Depth uncertainty model

Three independent error sources are propagated via:

```
σ_h² ≈ s² · σ_v² + (v_g − v_w)² · σ_s² + σ_g²
```

| Term | Symbol | Meaning |
|------|--------|---------|
| s² σ_v² | σ_v (px) | Waterline localisation error from mask edge quality |
| (v_g−v_w)² σ_s² | σ_s (cm/px) | Scale calibration error; grows with depth |
| σ_g² | σ_g (cm) | Ground-reference landmark uncertainty |

`σ_s` is the OLS standard error of the slope from `LinearCalib.sigma_s`.
Convert a ground-contact pixel error to cm with `σ_g = s · σ_g_px`.

```python
from floodcam.calib import depth_uncertainty_cm, LinearCalib

lc = LinearCalib().fit(calib_points)
sigma = depth_uncertainty_cm(
    s=lc.cm_per_pixel(),
    sigma_v=0.5,      # px
    v_g=v_g,
    v_w=v_w,
    sigma_s=lc.sigma_s,
    sigma_g=0.0,
)
```

---

## Known limitation: pole-only calibration and the half-plane water mask

On the synthetic renderer, the water mask is a full-width horizontal half-plane
(all pixels below the waterline row). Segmentation metrics on this mask are
trivial (near-100% IoU) because the boundary is axis-aligned and noise-free.
This is documented and expected; real masks will have boundary irregularities
that reduce IoU. See `docs/DECISIONS.md` for the rationale.

---

## Annotation CLI

### Headless (scripts, CI, tests)

```bash
python -m floodcam.calib.annotate \
    --points-json path/to/marks.json \
    --camera-id cam_01 \
    --output configs/cam_01.yaml \
    [--mode linear|homography]
```

**Linear JSON format** (2-element marks):
```json
{
    "reference_marks": [
        [356.2, 0.0],
        [333.1, 10.0],
        [309.1, 20.0],
        [284.4, 30.0],
        [258.9, 40.0]
    ],
    "roi_polygon": [[50, 150], [590, 150], [590, 460], [50, 460]]
}
```

**Homography JSON format** (4-element marks: image_u, image_v, plane_x_cm, height_cm):
```json
{
    "reference_marks": [
        [320.0, 356.2, 0.0, 0.0],
        [280.0, 340.0, -20.0, 5.0],
        ["...more marks..."]
    ]
}
```

### Interactive (OpenCV window)

```bash
python -m floodcam.calib.annotate \
    --image path/to/frame.jpg \
    --camera-id cam_01 \
    --output configs/cam_01.yaml
```

Controls:
- **Left-click** — add reference mark (terminal prompts for height in cm)
- **Right-click** — add ROI polygon vertex
- **`u`** — undo last reference mark
- **`s`** — fit calibration and save YAML
- **`q`** — quit without saving

---

## Fitted parameters stored in the camera YAML

After calibration, `configs/<camera_id>.yaml` contains under `calibration:`:

```yaml
calibration:
  mode: linear
  scale_cm_per_px: 0.41234     # |slope| in cm/pixel
  intercept_cm: 146.74         # b in h = a*v + b
  sigma_s: 0.00031             # OLS std error of slope (cm/px)
  reference_points:            # raw marks used for fitting
    - [356.2, 0.0]
    - [333.1, 10.0]
```

For homography mode, `homography_matrix` replaces `scale_cm_per_px` /
`intercept_cm`.

---

## Serialisation round-trip

Reconstruct a calibration object from a saved YAML without re-fitting:

```python
from floodcam.config import CameraConfig
from floodcam.calib import LinearCalib, HomographyCalib

cfg = CameraConfig.load("configs/cam_01.yaml")

if cfg.calibration.mode == "linear":
    lc = LinearCalib.from_config(cfg.calibration)
    depth = lc.height_at(v_waterline_row)
else:
    hc = HomographyCalib.from_config(cfg.calibration)
    depth = hc.height_at(u_pole_col, v_waterline_row)
```

---

## Recording protocol checklist

See `docs/RECORDING_PROTOCOL.md` for the full field recording instructions.
Key requirements for calibration:

1. Mount camera at a fixed position, level or with a small downward tilt.
2. Place a vertical pole with tape marks at known heights (0, 5, 10, 20, 30 cm).
3. Record 5–10 reference frames with the pole in clear view and no water.
4. For each frame, note the pixel row of each tape mark using the annotation CLI.
5. For homography mode, add marks at 2–3 horizontal offsets from the pole centre
   (e.g., −20, 0, +20 cm) at the same heights.
6. Validate: `scale_cm_per_px` should be in the range 0.1–2.0 cm/px for typical
   installations; `sigma_s` should be < 0.05 cm/px after a good annotation run.
