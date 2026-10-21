"""SegFormer binary water segmentation: model construction, checkpoints, backend.

torch and transformers are imported lazily so `import floodcam` works without the [ml]
extra. Output classes: 0 = background, 1 = water.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from floodcam.seg.datasets import IMAGENET_MEAN, IMAGENET_STD, file_sha256

if TYPE_CHECKING:
    from floodcam.config import ExperimentConfig

CKPT_META = "floodcam_meta.json"
CKPT_ENV = "FLOODCAM_CKPT_DIR"

# Hugging Face hub ids (resolved only when the user trains; nothing is fetched at import).
DEFAULT_INIT = {
    "segformer_b0": "nvidia/mit-b0",  # ImageNet-pretrained encoder, random decode head
    "segformer_b1": "nvidia/mit-b1",
}
# Architecture presets for init_weights == "random" and the smoke test.
_PRESETS = {
    "segformer_b0": dict(
        depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
        num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
    ),
    "segformer_b1": dict(
        depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
    ),
    "segformer_smoke": dict(
        depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32],
        num_attention_heads=[1, 1, 2, 2], decoder_hidden_size=16,
        sr_ratios=[8, 4, 2, 1], mlp_ratios=[2, 2, 2, 2],
    ),
}
ID2LABEL = {0: "background", 1: "water"}


class MLExtraMissing(ImportError):
    pass


def _require_ml():
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MLExtraMissing(
            'torch/transformers not installed. Run: pip install -e ".[ml]"'
        ) from exc
    return torch, transformers


def build_model(model_name: str, init_weights: str = "default"):
    """Return a SegformerForSemanticSegmentation with 2 labels.

    init_weights: "default" (ImageNet encoder for the model name), "random" (no weights,
    architecture preset only; also used by the smoke test), or any HF hub id / local path
    (e.g. an ADE-pretrained SegFormer checkpoint). Hub ids need network access.
    """
    _, tf = _require_ml()
    if model_name not in _PRESETS:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(_PRESETS)}")
    if init_weights == "random" or model_name == "segformer_smoke":
        cfg = tf.SegformerConfig(
            num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
            **_PRESETS[model_name],
        )
        return tf.SegformerForSemanticSegmentation(cfg)
    src = DEFAULT_INIT[model_name] if init_weights == "default" else init_weights
    return tf.SegformerForSemanticSegmentation.from_pretrained(
        src, num_labels=2, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}, ignore_mismatched_sizes=True,
    )


def checkpoint_sha256(ckpt_dir: Path) -> str:
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = Path(ckpt_dir) / name
        if p.is_file():
            return file_sha256(p)
    raise FileNotFoundError(f"no weight file in {ckpt_dir}")


def save_checkpoint(model, out_dir: Path, model_name: str, image_size: int, extra: dict) -> str:
    """Save HF weights + floodcam_meta.json. Returns the weight-file sha256."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    sha = checkpoint_sha256(out_dir)
    meta = {"model_name": model_name, "image_size": image_size, "sha256": sha, **extra}
    (out_dir / CKPT_META).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return sha


def resolve_ckpt_dir(ckpt_dir: Optional[Path | str] = None) -> Path:
    d = ckpt_dir or os.environ.get(CKPT_ENV)
    if not d:
        raise FileNotFoundError(f"no checkpoint given: pass ckpt_dir or set {CKPT_ENV}")
    d = Path(d)
    if not (d / CKPT_META).is_file() and (d / "best" / CKPT_META).is_file():
        d = d / "best"
    if not (d / CKPT_META).is_file():
        raise FileNotFoundError(f"{d} is not a floodcam checkpoint (missing {CKPT_META})")
    return d


def auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SegFormerBackend:
    """SegmentationBackend backed by a fine-tuned SegFormer checkpoint.

    The ROI is applied by the caller, not here. model_version = model name + first 8 hex
    chars of the checkpoint's SHA256.
    """

    def __init__(
        self,
        exp_cfg: Optional["ExperimentConfig"] = None,
        ckpt_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
    ) -> None:
        torch, tf = _require_ml()
        self._torch = torch
        d = resolve_ckpt_dir(ckpt_dir)
        meta = json.loads((d / CKPT_META).read_text(encoding="utf-8"))
        self.threshold = exp_cfg.seg_prob_threshold if exp_cfg is not None else 0.5
        self.image_size = int(meta["image_size"])
        self.device = device or auto_device(torch)
        self.model = tf.SegformerForSemanticSegmentation.from_pretrained(d).to(self.device).eval()
        self.checkpoint_sha256 = checkpoint_sha256(d)
        self.model_version = f"{meta['model_name']}-{self.checkpoint_sha256[:8]}"

    def predict_proba(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(H, W) float32 water probability at the input resolution."""
        import cv2

        torch = self._torch
        h, w = frame_rgb.shape[:2]
        s = self.image_size
        x = cv2.resize(frame_rgb, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=t).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
            prob = torch.softmax(logits, dim=1)[0, 1]
        return prob.cpu().numpy().astype(np.float32)

    def predict_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(H, W) uint8 mask, 0 or 255."""
        return ((self.predict_proba(frame_rgb) >= self.threshold) * 255).astype(np.uint8)
