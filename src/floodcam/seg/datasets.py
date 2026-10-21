"""Segmentation datasets: manifest rows, public-data adapters, binary-mask dataset.

No torch import here: the dataset returns numpy arrays and the training script converts
them. Everything is deterministic given (rows, seed, epoch).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from floodcam.seg.classmap import ClassMap, DatasetSpec, resolve_water_ids

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # constants of the pretrained encoder, not fitted here
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SegRow:
    """One image + (optional) mask. Paths are relative to the dataset root."""

    image_id: str
    mask_id: str  # "" => all-background mask (negatives without annotation)
    source: str
    split: str  # "train" | "val"
    is_negative: bool
    image_path: str
    mask_path: str
    scene_id: str = ""
    synthetic: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def write_seg_manifest(rows: Sequence[SegRow], water_ids: dict, path: Path) -> str:
    """Write a JSONL manifest (header line + rows). Returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_header": True, "water_ids": water_ids}, sort_keys=True)]
    lines += [json.dumps(r.to_dict(), sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_sha256(path)


def read_seg_manifest(path: Path) -> tuple[list[SegRow], dict]:
    rows: list[SegRow] = []
    water_ids: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("_header"):
            water_ids = d["water_ids"]
        else:
            rows.append(SegRow(**d))
    return rows, water_ids


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class DatasetLayoutError(RuntimeError):
    """Raised when expected files are missing under the dataset root."""


def adapt_dataset(spec: DatasetSpec, data_root: Path) -> tuple[list[SegRow], Optional[list[int]]]:
    """Build manifest rows for one dataset; verifies the class map against its label file."""
    ds_root = Path(data_root) / spec.root
    water = resolve_water_ids(spec, ds_root)  # raises ClassMapError on missing class
    rows: list[SegRow] = []
    for split, sdir in spec.split_dirs.items():
        img_dir = ds_root / spec.image_dir.format(split=sdir)
        msk_dir = ds_root / spec.mask_dir.format(split=sdir)
        if not img_dir.is_dir() or not msk_dir.is_dir():
            raise DatasetLayoutError(f"{spec.key}: missing {img_dir} or {msk_dir}")
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in spec.image_ext)
        if not images:
            raise DatasetLayoutError(f"{spec.key}: no images in {img_dir}")
        for img in images:
            msk = msk_dir / (img.stem + spec.mask_ext)
            if not msk.is_file():
                raise DatasetLayoutError(f"{spec.key}: no mask for {img.name} at {msk}")
            rows.append(
                SegRow(
                    image_id=f"{spec.key}/{img.stem}",
                    mask_id=f"{spec.key}/{msk.stem}",
                    source=spec.key,
                    split=split,
                    is_negative=False,
                    image_path=img.relative_to(data_root).as_posix(),
                    mask_path=msk.relative_to(data_root).as_posix(),
                )
            )
    return rows, water


def _hash_unit(key: str, seed: int) -> float:
    d = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def adapt_negatives(
    classmap: ClassMap, data_root: Path, seed: int = 42
) -> list[SegRow]:
    """Negatives: images/<sequence>/*.jpg, optional masks/<sequence>/*.png (empty if absent)."""
    neg = classmap.negatives
    root = Path(data_root) / neg.root
    img_root = root / "images"
    if not img_root.is_dir():
        raise DatasetLayoutError(f"negatives: missing {img_root}")
    rows: list[SegRow] = []
    for img in sorted(p for p in img_root.rglob("*") if p.suffix.lower() in neg.image_ext):
        rel = img.relative_to(img_root)
        seq = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        msk = root / "masks" / rel.with_suffix(".png")
        split = "val" if _hash_unit(seq, seed) < neg.val_fraction else "train"
        rows.append(
            SegRow(
                image_id=f"negatives/{rel.with_suffix('').as_posix()}",
                mask_id=f"negatives/{rel.with_suffix('').as_posix()}" if msk.is_file() else "",
                source="negatives",
                split=split,
                is_negative=True,
                image_path=img.relative_to(data_root).as_posix(),
                mask_path=msk.relative_to(data_root).as_posix() if msk.is_file() else "",
            )
        )
    return rows


def build_public_manifest(
    classmap: ClassMap, data_root: Path, seed: int = 42
) -> tuple[list[SegRow], dict]:
    rows: list[SegRow] = []
    water_ids: dict = {}
    for spec in classmap.enabled_datasets():
        r, water = adapt_dataset(spec, data_root)
        rows += r
        water_ids[spec.key] = water
    if classmap.negatives.enabled:
        rows += adapt_negatives(classmap, data_root, seed)
    ids = [r.image_id for r in rows]
    if len(ids) != len(set(ids)):
        raise DatasetLayoutError("duplicate image_id across datasets")
    return rows, water_ids


def write_public_splits(
    rows: Sequence[SegRow], manifest_sha: str, path: Path, seed: int = 42
) -> None:
    """IDs only. Public sets keep their official split; negatives are sequence-hashed."""
    sources: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        sources.setdefault(r.source, {"train": [], "val": []})[r.split].append(r.image_id)
    doc = {
        "seed": seed,
        "created": date.today().isoformat(),
        "type": "public_seg",
        "manifest_sha256": manifest_sha,
        "sources": sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mask binarisation, class weights, sampling weights
# ---------------------------------------------------------------------------


def binarize_mask(mask: np.ndarray, water_ids: Optional[Sequence[int]]) -> np.ndarray:
    """Return {0,1} uint8: pixels whose value is in water_ids (None => value > 0)."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    if water_ids is None:
        return (mask > 0).astype(np.uint8)
    return np.isin(mask, list(water_ids)).astype(np.uint8)


def load_binary_mask(
    root: Path, row: SegRow, water_ids: dict, shape: tuple[int, int]
) -> np.ndarray:
    import cv2

    if not row.mask_path:
        return np.zeros(shape, dtype=np.uint8)
    raw = cv2.imread(str(Path(root) / row.mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(Path(root) / row.mask_path)
    return binarize_mask(raw, water_ids.get(row.source))


def compute_class_weights(
    rows: Sequence[SegRow],
    root: Path,
    water_ids: dict,
    max_images: int = 2000,
    seed: int = 42,
    size: int = 128,
) -> list[float]:
    """Inverse-frequency class weights [w_background, w_water] from TRAIN rows only.

    Raises ValueError if any row is not in the train split (leakage guard).
    """
    import cv2

    bad = [r.image_id for r in rows if r.split != "train"]
    if bad:
        raise ValueError(f"class weights must use train rows only; got non-train: {bad[:3]}")
    if not rows:
        raise ValueError("no training rows")
    idx = np.arange(len(rows))
    if len(rows) > max_images:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_images, replace=False))
    counts = np.zeros(2, dtype=np.float64)
    for i in idx:
        r = rows[int(i)]
        img = cv2.imread(str(Path(root) / r.image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(Path(root) / r.image_path)
        m = load_binary_mask(root, r, water_ids, img.shape[:2])
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        counts[1] += m.sum()
        counts[0] += m.size - m.sum()
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return [float(w[0]), float(w[1])]


def sampling_weights(rows: Sequence[SegRow], negative_weight: float) -> list[float]:
    return [negative_weight if r.is_negative else 1.0 for r in rows]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SegDataset:
    """Map-style dataset yielding (float32 CHW normalised image, int64 HW mask in {0,1})."""

    def __init__(
        self,
        rows: Sequence[SegRow],
        root: Path,
        water_ids: dict,
        image_size: int,
        train: bool = False,
        aug_cfg: Optional[object] = None,
        seed: int = 42,
    ) -> None:
        self.rows = list(rows)
        self.root = Path(root)
        self.water_ids = water_ids
        self.image_size = image_size
        self.train = train
        self.aug_cfg = aug_cfg
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def load_raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Full-resolution RGB uint8 image and {0,1} uint8 mask (no augmentation)."""
        import cv2

        r = self.rows[i]
        bgr = cv2.imread(str(self.root / r.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r.image_path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = load_binary_mask(self.root, r, self.water_ids, img.shape[:2])
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from floodcam.seg.augment import augment

        img, mask = self.load_raw(i)
        if self.train and self.aug_cfg is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img, mask = augment(img, mask, rng, self.aug_cfg)  # type: ignore[arg-type]
        s = self.image_size
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return x.transpose(2, 0, 1).copy(), mask.astype(np.int64)
