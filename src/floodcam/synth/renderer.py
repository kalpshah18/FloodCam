"""Pinhole-camera synthetic scene renderer.

Geometry convention
-------------------
World: X right, Y up (height), Z forward.
Camera at world position (0, H, 0), tilted down by depression angle alpha.

Camera-frame coordinates for world point (Xw, Yw, Zw):
    X_c =  Xw
    Y_c =  (H - Yw)*cos(alpha) - Zw*sin(alpha)   [+ => below optical axis => larger row]
    Z_c =  (H - Yw)*sin(alpha) + Zw*cos(alpha)   [+ => in front of camera]

Projection (principal point at image centre):
    u = cx_img + f * X_c / Z_c
    v = cy_img + f * Y_c / Z_c

Derived formula (pole at horizontal distance D, point at height h):
    dv/dh = -f * D / Z_c(h)^2
    cm_per_pixel(h) = Z_c(h)^2 / (f * D)

where Z_c(h) = (H - h)*sin(alpha) + D*cos(alpha).
Horizon row: v_horizon = cy_img - f*tan(alpha).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SceneConfig:
    """Camera geometry and scene layout parameters."""

    camera_height_cm: float = 120.0
    camera_tilt_deg: float = 20.0  # depression angle (positive = looking down)
    focal_length_px: float = 600.0
    img_width: int = 640
    img_height: int = 480
    pole_distance_cm: float = 200.0  # horizontal distance to pole
    pole_height_cm: float = 50.0  # visible pole height above ground
    pole_width_px: int = 6
    pole_x_frac: float = 0.5  # fractional column of pole centre
    calib_heights_cm: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0, 30.0, 40.0])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class SynthFrame:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    mask: np.ndarray  # (H, W) uint8 {0, 255}
    depth_cm: float  # true water depth
    waterline_row: float  # analytic waterline row (float)
    ground_contact_row: float  # analytic row where pole meets ground
    cm_per_pixel: float  # at pole, near depth_cm
    calib_points: list[tuple[float, float]]  # [(row, height_cm), ...]
    variant: str
    synthetic: bool = True


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class SynthRenderer:
    """Deterministic pinhole-camera scene renderer.

    Same seed + same depth + same frame_idx => byte-identical output.
    """

    def __init__(self, config: SceneConfig, seed: int = 42) -> None:
        self._cfg = config
        self._seed = seed
        self._alpha = math.radians(config.camera_tilt_deg)
        self._H = config.camera_height_cm
        self._D = config.pole_distance_cm
        self._f = config.focal_length_px
        self._cx_pole = int(config.img_width * config.pole_x_frac)
        self._cy = config.img_height // 2
        self._v_horizon = self._cy - self._f * math.tan(self._alpha)

    # ------------------------------------------------------------------ #
    # Geometry API
    # ------------------------------------------------------------------ #

    def project_height(self, h_cm: float) -> tuple[float, float]:
        """Return (image_row_v, Z_c) for a point at height h_cm on the pole."""
        Y_c = (self._H - h_cm) * math.cos(self._alpha) - self._D * math.sin(self._alpha)
        Z_c = (self._H - h_cm) * math.sin(self._alpha) + self._D * math.cos(self._alpha)
        v = self._cy + self._f * Y_c / Z_c
        return v, Z_c

    def cm_per_pixel(self, h_cm: float = 0.0) -> float:
        """Scale factor cm/px at height h_cm: Z_c(h)^2 / (f * D)."""
        _, Z_c = self.project_height(h_cm)
        return Z_c**2 / (self._f * self._D)

    def get_calib_points(self) -> list[tuple[float, float]]:
        """Return [(row, height_cm)] for each calibration height."""
        return [(self.project_height(h)[0], h) for h in self._cfg.calib_heights_cm]

    # ------------------------------------------------------------------ #
    # Main rendering entry point
    # ------------------------------------------------------------------ #

    def render_frame(
        self,
        depth_cm: float,
        frame_idx: int = 0,
        variant: str = "normal",
    ) -> SynthFrame:
        """Render one frame.

        Args:
            depth_cm: True water depth (0 = dry).
            frame_idx: Per-depth frame counter; shifts the RNG seed so
                       consecutive frames at the same depth differ slightly.
            variant: "normal" | "reflection" | "wet_road" | "puddle".

        Returns:
            SynthFrame with image, mask, and analytic ground-truth values.
        """
        rng = np.random.default_rng([self._seed, int(depth_cm * 100) & 0xFFFF, frame_idx])

        v_g, _ = self.project_height(0.0)
        v_w, _ = self.project_height(depth_cm)
        s = self.cm_per_pixel(depth_cm)

        img = np.zeros((self._cfg.img_height, self._cfg.img_width, 3), dtype=np.uint8)
        mask = np.zeros((self._cfg.img_height, self._cfg.img_width), dtype=np.uint8)

        self._render_background(img, rng, variant)
        self._render_pole(img)

        if variant == "normal" and depth_cm > 0:
            self._render_water(img, mask, v_w, rng)
        elif variant == "reflection":
            self._render_reflection(img, v_g, rng)
        elif variant == "wet_road":
            pass  # dark background already set in _render_background
        elif variant == "puddle":
            self._render_puddle(img, mask, v_g, rng)

        return SynthFrame(
            image=img,
            mask=mask,
            depth_cm=depth_cm,
            waterline_row=v_w,
            ground_contact_row=v_g,
            cm_per_pixel=s,
            calib_points=self.get_calib_points(),
            variant=variant,
        )

    # ------------------------------------------------------------------ #
    # Private rendering helpers
    # ------------------------------------------------------------------ #

    def _render_background(self, img: np.ndarray, rng: np.random.Generator, variant: str) -> None:
        H, W = img.shape[:2]
        v_hor = int(np.clip(self._v_horizon, 0, H - 1))

        # Sky: light-blue gradient, vectorised across columns
        if v_hor > 0:
            t = np.linspace(0.0, 1.0, v_hor)[:, None]  # (v_hor, 1)
            sky = np.empty((v_hor, 1, 3), np.uint8)
            sky[..., 0] = np.clip(135 + 50 * t, 0, 255).astype(np.uint8)
            sky[..., 1] = np.clip(180 + 30 * t, 0, 255).astype(np.uint8)
            sky[..., 2] = np.clip(220 - 20 * t, 0, 255).astype(np.uint8)
            img[:v_hor] = sky
            noise = rng.integers(-15, 16, (v_hor, W, 3), dtype=np.int16)
            img[:v_hor] = np.clip(img[:v_hor].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Ground: brownish, slightly lighter toward horizon
        ground_rows = H - v_hor
        if ground_rows > 0:
            if variant == "wet_road":
                base = np.array([70, 70, 70], dtype=np.float32)
            else:
                base = np.array(
                    [
                        110 + rng.integers(-10, 11),
                        95 + rng.integers(-10, 11),
                        75 + rng.integers(-10, 11),
                    ],
                    dtype=np.float32,
                )
            t_g = np.linspace(1.0, 0.7, ground_rows)[:, None, None]  # brighter near horizon
            ground = np.clip(base[None, None, :] * t_g, 0, 255).astype(np.uint8)
            img[v_hor:] = ground
            noise = rng.integers(-10, 11, (ground_rows, W, 3), dtype=np.int16)
            img[v_hor:] = np.clip(img[v_hor:].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _render_pole(self, img: np.ndarray) -> None:
        H, W = img.shape[:2]
        cx = self._cx_pole
        hw = self._cfg.pole_width_px // 2

        v_top, _ = self.project_height(self._cfg.pole_height_cm)
        v_bot, _ = self.project_height(0.0)
        r_top = int(np.clip(math.floor(v_top), 0, H - 1))
        r_bot = int(np.clip(math.ceil(v_bot), 0, H - 1))
        c_l = max(0, cx - hw)
        c_r = min(W, cx + hw + 1)

        img[r_top : r_bot + 1, c_l:c_r] = [230, 230, 230]

        # Tick marks every 5 cm
        h = 0.0
        while h <= self._cfg.pole_height_cm + 0.01:
            v_tick, _ = self.project_height(h)
            r = int(round(v_tick))
            if 0 <= r < H:
                img[r, max(0, c_l - 4) : min(W, c_r + 4)] = [30, 30, 30]
            h += 5.0

    def _render_water(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        v_w: float,
        rng: np.random.Generator,
    ) -> None:
        H, W = img.shape[:2]
        cx = self._cx_pole
        hw = self._cfg.pole_width_px // 2 + 4  # exclude pole columns from mask
        r_w = int(np.clip(math.ceil(v_w), 0, H - 1))

        # Water colour
        wc = np.array(
            [
                40.0 + rng.uniform(-10, 10),
                100.0 + rng.uniform(-15, 15),
                160.0 + rng.uniform(-15, 15),
            ]
        )
        wc = np.clip(wc, 0, 200).astype(np.float32)

        region = img[r_w:].astype(np.float32)
        img[r_w:] = np.clip(region * 0.4 + wc * 0.6, 0, 255).astype(np.uint8)

        # Mask: all columns except pole
        mask[r_w:, : max(0, cx - hw)] = 255
        mask[r_w:, min(W, cx + hw + 1) :] = 255

        # Specular highlights
        n_hl = int(rng.integers(4, 9))
        for _ in range(n_hl):
            hr = int(rng.integers(r_w, H))
            hc = int(rng.integers(0, W))
            if abs(hc - cx) > hw + 2:
                s = int(rng.integers(1, 4))
                patch = img[max(0, hr - s) : hr + s + 1, max(0, hc - s) : hc + s + 1]
                img[max(0, hr - s) : hr + s + 1, max(0, hc - s) : hc + s + 1] = np.clip(
                    patch.astype(np.int16) + 60, 0, 255
                ).astype(np.uint8)

    def _render_reflection(self, img: np.ndarray, v_g: float, rng: np.random.Generator) -> None:
        H, W = img.shape[:2]
        r_g = int(np.clip(v_g, 0, H - 1))
        top = max(0, r_g - int(rng.integers(30, 80)))
        bot = min(H, top + int(rng.integers(20, 50)))
        left = int(rng.integers(0, W // 3))
        right = int(rng.integers(2 * W // 3, W))

        refl = np.array([60.0, 120.0, 180.0])
        alpha = rng.uniform(0.3, 0.6)
        patch = img[top:bot, left:right].astype(np.float32)
        img[top:bot, left:right] = np.clip(patch * (1 - alpha) + refl * alpha, 0, 255).astype(
            np.uint8
        )

    def _render_puddle(
        self, img: np.ndarray, mask: np.ndarray, v_g: float, rng: np.random.Generator
    ) -> None:
        H, W = img.shape[:2]
        r_g = int(np.clip(v_g, 0, H - 1))
        cy_p = int(rng.integers(max(r_g - 20, H // 2), H - 10))
        cx_p = int(rng.integers(10, W // 4))
        r_p = int(rng.integers(10, 30))

        Y_idx, X_idx = np.ogrid[:H, :W]
        circle = (Y_idx - cy_p) ** 2 + (X_idx - cx_p) ** 2 <= r_p**2
        wc = np.array([50.0, 110.0, 170.0])
        img[circle] = np.clip(img[circle].astype(np.float32) * 0.4 + wc * 0.6, 0, 255).astype(
            np.uint8
        )
        mask[circle] = 255
