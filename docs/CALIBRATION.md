# Camera Calibration Guide

> Full documentation will be written at milestone 3.

## Overview

FloodCam uses geometric depth estimation. Given a reference object of known height
(a marked pole, curb marking, or wall stripe), the waterline row in the frame is
converted to a depth estimate using the scale factor `s` (cm/pixel).

**Depth formula:** `h = s * (v_g - v_w)`

- `v_g` — ground-contact row of the reference object (pixels from top of frame)
- `v_w` — waterline row at the reference object (pixels from top)
- `s` — scale factor in cm/pixel, fitted from reference points of known height

## Calibration modes

- **`linear`** — assumes the camera is roughly level. Fit `s` from at least
  two reference points at known heights on the pole. Suitable when camera tilt
  is less than ~15 degrees.
- **`homography`** — corrects for camera tilt by mapping the ground plane to a
  rectified view. Requires at least 4 ground-plane point correspondences.

## CLI tool (available after milestone 3)

```bash
python -m floodcam.calib.click_points --camera configs/my_cam.yaml --frame path/to/frame.jpg
```

Saves fitted parameters back into the camera YAML under the `calibration:` key.

## Uncertainty model

`sigma_h^2 ≈ s^2 * sigma_v^2 + (v_g - v_w)^2 * sigma_s^2 + sigma_g^2`

- `sigma_v` — waterline localization error in px (from mask edge quality)
- `sigma_s` — scale-fit error (from reference point placement)
- `sigma_g` — ground-reference error (landmark stability)

See `docs/DECISIONS.md` and `scripts/depth_error_analysis.py` for empirical validation.
