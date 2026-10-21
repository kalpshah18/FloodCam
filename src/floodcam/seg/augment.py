"""Training augmentations (numpy + OpenCV only; deterministic given the rng).

Horizontal flips only: a vertical flip would invert the gravity prior (water lies below the
waterline), which the geometry in this project relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugConfig:
    hflip_p: float = 0.5
    crop_min_frac: float = 0.6  # random crop side as a fraction of the image side
    crop_p: float = 0.8
    jitter_p: float = 0.8
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue_shift: int = 6  # OpenCV hue units (0-179)
    blur_p: float = 0.2
    noise_p: float = 0.3
    noise_std: float = 6.0
    jpeg_p: float = 0.3
    jpeg_quality_range: tuple[int, int] = (30, 90)
    rain_p: float = 0.2


def random_crop_resize(img: np.ndarray, mask: np.ndarray, rng, min_frac: float):
    import cv2

    h, w = mask.shape
    fh, fw = rng.uniform(min_frac, 1.0, size=2)
    ch, cw = max(8, int(h * fh)), max(8, int(w * fw))
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    img_c = cv2.resize(img[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask_c = cv2.resize(mask[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    return img_c, mask_c


def color_jitter(img: np.ndarray, rng, cfg: AugConfig) -> np.ndarray:
    import cv2

    x = img.astype(np.float32)
    x = x * (1.0 + rng.uniform(-cfg.brightness, cfg.brightness))
    mean = x.mean()
    x = (x - mean) * (1.0 + rng.uniform(-cfg.contrast, cfg.contrast)) + mean
    x = np.clip(x, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + rng.uniform(-cfg.saturation, cfg.saturation)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_shift, cfg.hue_shift)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def gaussian_blur(img: np.ndarray, rng) -> np.ndarray:
    import cv2

    k = int(rng.choice([3, 5, 7]))
    return cv2.GaussianBlur(img, (k, k), 0)


def gaussian_noise(img: np.ndarray, rng, std: float) -> np.ndarray:
    noise = rng.normal(0.0, std, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, rng, q_range: tuple[int, int]) -> np.ndarray:
    import cv2

    q = int(rng.integers(q_range[0], q_range[1] + 1))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def rain_streaks(img: np.ndarray, rng) -> np.ndarray:
    """Overlay thin, slanted, semi-transparent light streaks."""
    import cv2

    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    n = int(rng.integers(max(10, w // 8), max(20, w // 3)))
    slant = rng.uniform(-0.3, 0.3)
    for _ in range(n):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(max(4, h // 20), max(8, h // 6)))
        x1 = int(x0 + slant * length)
        y1 = y0 + length
        cv2.line(layer, (x0, y0), (x1, y1), (220, 220, 230), 1)
    alpha = rng.uniform(0.15, 0.4)
    out = img.astype(np.float32) * (1 - alpha * (layer.max(axis=2, keepdims=True) > 0)) + (
        layer.astype(np.float32) * alpha
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, mask: np.ndarray, rng, cfg: AugConfig) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full pipeline. Geometric ops hit both; photometric ops only the image."""
    if rng.random() < cfg.hflip_p:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
    if rng.random() < cfg.crop_p:
        img, mask = random_crop_resize(img, mask, rng, cfg.crop_min_frac)
    if rng.random() < cfg.jitter_p:
        img = color_jitter(img, rng, cfg)
    if rng.random() < cfg.blur_p:
        img = gaussian_blur(img, rng)
    if rng.random() < cfg.rain_p:
        img = rain_streaks(img, rng)
    if rng.random() < cfg.noise_p:
        img = gaussian_noise(img, rng, cfg.noise_std)
    if rng.random() < cfg.jpeg_p:
        img = jpeg_compress(img, rng, cfg.jpeg_quality_range)
    return img, mask
