"""Config-driven source-class -> binary-water mapping.

Class names are never trusted from memory: every mapped name/id is verified against the
dataset's own label file, and a missing one raises ClassMapError.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class ClassMapError(ValueError):
    """Raised when a class map is malformed or names a class absent from the label file."""


@dataclass
class DatasetSpec:
    key: str
    enabled: bool
    root: str
    label_file: Optional[str]
    label_format: str
    water_class_names: list[str] = field(default_factory=list)
    water_class_ids: list[int] = field(default_factory=list)
    positive_values: list[int] = field(default_factory=list)
    id_offset: int = 1
    image_dir: str = "images/{split}"
    mask_dir: str = "masks/{split}"
    image_ext: list[str] = field(default_factory=lambda: [".jpg", ".png"])
    mask_ext: str = ".png"
    split_dirs: dict[str, str] = field(default_factory=lambda: {"train": "train", "val": "val"})
    license_note: str = "UNVERIFIED"
    license_verified: bool = False
    domain_gap_note: str = ""


@dataclass
class NegativesSpec:
    enabled: bool = True
    root: str = "negatives"
    sampling_weight: float = 2.0
    val_fraction: float = 0.2
    image_ext: list[str] = field(default_factory=lambda: [".jpg", ".png"])


@dataclass
class ClassMap:
    datasets: dict[str, DatasetSpec]
    negatives: NegativesSpec

    def enabled_datasets(self) -> list[DatasetSpec]:
        return [d for d in self.datasets.values() if d.enabled]


_FORMATS = {"ade_objectinfo_csv", "id_name_txt", "coco_categories_json", "binary"}


def load_classmap(path: Path | str) -> ClassMap:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "datasets" not in raw:
        raise ClassMapError(f"{path}: missing 'datasets' section")
    specs: dict[str, DatasetSpec] = {}
    for key, d in raw["datasets"].items():
        fmt = d.get("label_format")
        if fmt not in _FORMATS:
            raise ClassMapError(f"{key}: label_format {fmt!r} not in {sorted(_FORMATS)}")
        layout = d.get("layout", {})
        spec = DatasetSpec(
            key=key,
            enabled=bool(d.get("enabled", False)),
            root=d.get("root", key),
            label_file=d.get("label_file"),
            label_format=fmt,
            water_class_names=list(d.get("water_class_names", []) or []),
            water_class_ids=[int(i) for i in d.get("water_class_ids", []) or []],
            positive_values=[int(i) for i in d.get("positive_values", []) or []],
            id_offset=int(d.get("id_offset", 1)),
            image_dir=layout.get("images", "images/{split}"),
            mask_dir=layout.get("masks", "masks/{split}"),
            image_ext=list(layout.get("image_ext", [".jpg", ".png"])),
            mask_ext=layout.get("mask_ext", ".png"),
            split_dirs=dict(d.get("split_dirs", {"train": "train", "val": "val"})),
            license_note=d.get("license_note", "UNVERIFIED"),
            license_verified=bool(d.get("license_verified", False)),
            domain_gap_note=d.get("domain_gap_note", ""),
        )
        if fmt != "binary" and not (spec.water_class_names or spec.water_class_ids):
            raise ClassMapError(f"{key}: no water_class_names / water_class_ids given")
        if fmt != "binary" and not spec.label_file:
            raise ClassMapError(f"{key}: label_format {fmt} requires label_file")
        specs[key] = spec
    n = raw.get("negatives", {}) or {}
    neg = NegativesSpec(**{k: v for k, v in n.items() if k in NegativesSpec.__dataclass_fields__})
    return ClassMap(datasets=specs, negatives=neg)


# ---------------------------------------------------------------------------
# Label-file parsers: return {class_id: [names]} read from the dataset's own file
# ---------------------------------------------------------------------------


def parse_labels(spec: DatasetSpec, dataset_root: Path) -> dict[int, list[str]]:
    if spec.label_format == "binary":
        return {}
    path = dataset_root / str(spec.label_file)
    if not path.is_file():
        raise ClassMapError(f"{spec.key}: label file not found: {path}")
    if spec.label_format == "ade_objectinfo_csv":
        return _parse_ade_csv(path, spec)
    if spec.label_format == "id_name_txt":
        return _parse_id_name_txt(path)
    return _parse_coco_json(path)


def _parse_ade_csv(path: Path, spec: DatasetSpec) -> dict[int, list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        if "name" not in header:
            raise ClassMapError(f"{path}: no 'Name' column (found {list(header)})")
        out: dict[int, list[str]] = {}
        for i, row in enumerate(reader):
            names = [t.strip() for t in row[header["name"]].split(";") if t.strip()]
            cid = int(row[header["idx"]]) if "idx" in header else i + spec.id_offset
            out[cid] = names
    return out


_ID_NAME = re.compile(r"^\s*(\d+)\s*[:,\s]\s*(.+?)\s*$")


def _parse_id_name_txt(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ID_NAME.match(line)
        if m:
            out[int(m.group(1))] = [m.group(2)]
    if not out:
        raise ClassMapError(f"{path}: no '<id>: <name>' lines parsed")
    return out


def _parse_coco_json(path: Path) -> dict[int, list[str]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    cats = data.get("categories") if isinstance(data, dict) else None
    if not cats:
        raise ClassMapError(f"{path}: no 'categories' list")
    return {int(c["id"]): [str(c["name"])] for c in cats}


def resolve_water_ids(spec: DatasetSpec, dataset_root: Path) -> Optional[list[int]]:
    """Resolve the configured water classes to mask pixel values, verifying each exists.

    Returns None for `binary` datasets with no explicit positive_values (rule: mask > 0).
    """
    if spec.label_format == "binary":
        return list(spec.positive_values) or None
    labels = parse_labels(spec, dataset_root)
    by_name: dict[str, int] = {}
    for cid, names in labels.items():
        for n in names:
            by_name.setdefault(n.lower(), cid)
        if len(names) > 1:
            by_name.setdefault(";".join(names).lower(), cid)
    ids: list[int] = []
    for name in spec.water_class_names:
        if name.lower() not in by_name:
            raise ClassMapError(
                f"{spec.key}: mapped class name {name!r} not found in {spec.label_file} "
                f"({len(labels)} classes parsed)"
            )
        ids.append(by_name[name.lower()])
    for cid in spec.water_class_ids:
        if cid not in labels:
            raise ClassMapError(f"{spec.key}: mapped class id {cid} not in {spec.label_file}")
        ids.append(cid)
    return sorted(set(ids))
