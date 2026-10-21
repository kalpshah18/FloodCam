"""SegFormer training (pretrain on public data, finetune on self-labeled frames).

Importable and torch-lazy at module level: the guards and config handling below run
without the [ml] extra; only `run_training` needs torch.

Guards (fail loudly, never silently skip):
  * finetune refuses synthetic rows unless allow_synthetic (--smoke-synthetic)
  * finetune refuses any scene named as the locked final-test scene
  * class weights are computed from the training split only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml

from floodcam.seg.augment import AugConfig
from floodcam.seg.datasets import (
    SegDataset,
    SegRow,
    compute_class_weights,
    file_sha256,
    read_seg_manifest,
    sampling_weights,
)
from floodcam.seg.metrics import ConfusionCounts

ROOT = Path(__file__).resolve().parents[3]


class TrainGuardError(RuntimeError):
    """A safety rule (locked scene, synthetic data, leakage) was violated."""


@dataclass
class SegTrainConfig:
    stage: str = "pretrain"  # "pretrain" | "finetune"
    model: str = "segformer_b0"  # segformer_b0 | segformer_b1 | segformer_smoke
    init_weights: str = "default"  # default | random | <hf id or local path>
    image_size: int = 512
    epochs: int = 40
    batch_size: int = 16
    lr: float = 6e-5
    weight_decay: float = 1e-2
    seed: int = 42
    num_workers: int = 4
    amp: bool = True
    patience: int = 6
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    negative_weight: Optional[float] = None  # None => take sampling_weight from the classmap
    class_weight_max_images: int = 2000
    class_map: str = "configs/seg_classmap.yaml"
    data_root: str = "data/public"
    manifest: str = "data/public/seg_manifest.jsonl"
    splits_file: str = ""  # finetune: scene-level split (coarse_train_scenes/coarse_val_scenes)
    locked_scenes: list[str] = field(default_factory=list)
    aug: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "SegTrainConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def load_locked_scenes(splits_dir: Path, extra: Sequence[str] = ()) -> set[str]:
    """Every scene named as locked in any data/splits/*.json, plus explicit extras."""
    locked = set(extra)
    for p in sorted(Path(splits_dir).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        v = d.get("locked_test_scene") if isinstance(d, dict) else None
        if isinstance(v, str):
            locked.add(v)
        elif isinstance(v, list):
            locked.update(v)
    return locked


def guard_finetune(
    rows: Sequence[SegRow],
    train_scenes: Sequence[str],
    val_scenes: Sequence[str],
    locked: set[str],
    allow_synthetic: bool,
) -> None:
    used = set(train_scenes) | set(val_scenes) | {r.scene_id for r in rows}
    hit = used & locked
    if hit:
        raise TrainGuardError(f"refusing locked final-test scene(s): {sorted(hit)}")
    overlap = set(train_scenes) & set(val_scenes)
    if overlap:
        raise TrainGuardError(f"scene(s) in both train and val: {sorted(overlap)}")
    if any(r.synthetic for r in rows) and not allow_synthetic:
        raise TrainGuardError(
            "finetune refuses synthetic data; pass --smoke-synthetic for a pipeline dry run"
        )


def guard_pretrain(rows: Sequence[SegRow], smoke: bool) -> None:
    if any(r.synthetic for r in rows) and not smoke:
        raise TrainGuardError("pretrain on synthetic data is only allowed with --smoke")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def rows_from_scene_manifest(manifest_path: Path, root: Path) -> list[SegRow]:
    """Convert a scene manifest (ManifestRow JSONL) into SegRows (paths relative to root)."""
    from floodcam.io.manifest import read_jsonl

    out: list[SegRow] = []
    for m in read_jsonl(Path(manifest_path)):
        if not m.mask_path and not m.is_negative:
            raise TrainGuardError(f"{m.frame_id}: no mask_path; finetune needs labeled frames")
        out.append(
            SegRow(
                image_id=m.frame_id, mask_id=m.frame_id if m.mask_path else "",
                source="synthetic" if m.synthetic else "self_labeled", split="",
                is_negative=m.is_negative, image_path=m.image_path, mask_path=m.mask_path,
                scene_id=m.scene_id, synthetic=m.synthetic,
            )
        )
    return out


def make_smoke_data(out_dir: Path) -> tuple[list[SegRow], list[SegRow]]:
    """Render a handful of synthetic frames (2 scenes). Pipeline checks only."""
    import cv2

    from floodcam.synth.renderer import SceneConfig, SynthRenderer

    out_dir.mkdir(parents=True, exist_ok=True)
    train: list[SegRow] = []
    val: list[SegRow] = []
    scenes = [
        ("smoke_scene_a", SceneConfig(camera_tilt_deg=20.0), train, "train", [5.0, 10.0, 20.0]),
        ("smoke_scene_b", SceneConfig(camera_tilt_deg=25.0, camera_height_cm=110.0), val, "val",
         [5.0, 15.0]),
    ]
    for si, (sid, scfg, dest, split, depths) in enumerate(scenes):
        rend = SynthRenderer(scfg, seed=1000 + si)
        for di, d in enumerate(depths):
            for fi in range(2):
                f = rend.render_frame(d, frame_idx=fi)
                name = f"{sid}_{di}_{fi}"
                cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{name}_mask.png"), f.mask)
                dest.append(
                    SegRow(
                        image_id=name, mask_id=name + "_mask", source="synthetic", split=split,
                        is_negative=False, image_path=f"{name}.png",
                        mask_path=f"{name}_mask.png", scene_id=sid, synthetic=True,
                    )
                )
    return train, val


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=ROOT,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _hardware(torch) -> dict:
    hw = {"platform": platform.platform(), "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
        hw["cuda"] = torch.version.cuda
    else:
        hw["gpu"] = None
    return hw


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _dice_loss(logits, target, eps: float = 1.0):
    import torch

    p = torch.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def run_training(
    cfg: SegTrainConfig,
    train_rows: Sequence[SegRow],
    val_rows: Sequence[SegRow],
    root: Path,
    water_ids: dict,
    run_id: str,
    output_dir: Path,
    results_dir: Path,
    manifest_sha: str,
    synthetic: bool,
    smoke: bool = False,
    max_steps: Optional[int] = None,
    negative_weight: float = 1.0,
    extra_meta: Optional[dict] = None,
) -> dict:

    from floodcam.seed import seed_everything
    from floodcam.seg.segformer import _require_ml, auto_device, build_model, save_checkpoint

    torch, transformers = _require_ml()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not val_rows:
        raise TrainGuardError("empty validation split")
    seed_everything(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = auto_device(torch)
    use_amp = cfg.amp and device == "cuda"

    run_dir = Path(results_dir) / run_id
    ckpt_root = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cw = compute_class_weights(
        train_rows, root, water_ids, cfg.class_weight_max_images, cfg.seed
    )
    aug = AugConfig(**cfg.aug)
    ds_tr = SegDataset(train_rows, root, water_ids, cfg.image_size, True, aug, cfg.seed)
    ds_va = SegDataset(val_rows, root, water_ids, cfg.image_size, False, None, cfg.seed)

    g = torch.Generator().manual_seed(cfg.seed)
    if negative_weight != 1.0 and any(r.is_negative for r in train_rows):
        sampler = WeightedRandomSampler(
            sampling_weights(train_rows, negative_weight), len(train_rows), generator=g
        )
        dl_tr = DataLoader(ds_tr, cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        dl_tr = DataLoader(
            ds_tr, cfg.batch_size, shuffle=True, generator=g, num_workers=cfg.num_workers
        )
    dl_va = DataLoader(ds_va, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg.model, cfg.init_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))

    def up(logits, size):
        return torch.nn.functional.interpolate(
            logits.float(), size=size, mode="bilinear", align_corners=False
        )

    def loss_fn(logits, y):
        lg = up(logits, y.shape[-2:])
        return cfg.ce_weight * ce(lg, y) + cfg.dice_weight * _dice_loss(lg, y)

    def evaluate():
        model.eval()
        cc, tot, n = ConfusionCounts(), 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                lg = model(pixel_values=x).logits
                tot += loss_fn(lg, y).item() * x.shape[0]
                n += x.shape[0]
                pred = up(lg, y.shape[-2:]).argmax(1)
                cc.update(pred.cpu().numpy(), y.cpu().numpy())
        return tot / max(n, 1), cc

    curves: list[dict] = []
    best_iou, best_epoch, bad = -1.0, -1, 0
    best_sha = last_sha = ""
    best_counts: dict = {}
    step = 0
    t0 = time.time()
    save_extra = {"stage": cfg.stage, "synthetic": synthetic, "init_weights": cfg.init_weights}
    for epoch in range(cfg.epochs):
        ds_tr.set_epoch(epoch)
        model.train()
        run_loss, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    lg = model(pixel_values=x).logits
                loss = loss_fn(lg, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(pixel_values=x).logits, y)
                loss.backward()
                opt.step()
            run_loss += loss.item()
            nb += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        sched.step()
        val_loss, cc = evaluate()
        iou, f1 = cc.iou(), cc.f1()
        curves.append({
            "epoch": epoch, "steps": step, "train_loss": run_loss / max(nb, 1),
            "val_loss": val_loss, "val_iou": iou, "val_f1": f1,
            "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2),
        })
        last_sha = save_checkpoint(model, ckpt_root / "last", cfg.model, cfg.image_size, save_extra)
        score = iou if iou is not None else -1.0
        if best_epoch < 0 or score > best_iou:
            best_iou, best_epoch, bad = score, epoch, 0
            best_sha = save_checkpoint(
                model, ckpt_root / "best", cfg.model, cfg.image_size, save_extra
            )
            best_counts = cc.to_dict()
        else:
            bad += 1
        if max_steps is not None and step >= max_steps:
            break
        if bad >= cfg.patience:
            break

    with (run_dir / "curves.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)

    metrics = {
        "run_id": run_id, "stage": cfg.stage, "synthetic": synthetic, "smoke": smoke,
        "note": "SYNTHETIC - pipeline validation only; not a result." if synthetic else "",
        "best_epoch": best_epoch, "best_val_iou": best_iou if best_iou >= 0 else None,
        "best_val_counts": best_counts, "epochs_run": len(curves), "steps": step,
        "val_resolution": cfg.image_size, "class_weights_train_only": cw,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "checkpoint_best": str(ckpt_root / "best"), "checkpoint_last": str(ckpt_root / "last"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    meta = {
        "run_id": run_id, "config": asdict(cfg), "git_commit": _git_commit(),
        "dataset_manifest_sha256": manifest_sha, "seed": cfg.seed,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "hardware": _hardware(torch), "device": device, "amp": use_amp,
        "init_weights": cfg.init_weights, "synthetic": synthetic, "smoke": smoke,
        "checkpoint_best_sha256": best_sha, "checkpoint_last_sha256": last_sha,
        "locked_scene_touched": False, "water_ids": water_ids,
        **(extra_meta or {}),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train SegFormer (binary water).")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "seg_pretrain.yaml")
    ap.add_argument("--stage", choices=["pretrain", "finetune"])
    ap.add_argument("--run-id")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--splits-file", type=Path)
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--locked-scene", action="append", default=[])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random SegFormer, rendered synthetic images, 2 steps, no downloads")
    ap.add_argument("--smoke-synthetic", action="store_true",
                    help="allow synthetic rows in the finetune stage (dry runs only)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    cfg = SegTrainConfig.load(a.config) if a.config.is_file() else SegTrainConfig()
    if a.stage:
        cfg.stage = a.stage
    run_id = a.run_id or f"seg-{cfg.stage}{'-smoke' if a.smoke else ''}-001"
    if (a.results_dir / run_id / "metrics.json").exists() and not a.overwrite:
        print(f"error: results/{run_id} exists; use --overwrite or a new --run-id", file=sys.stderr)
        return 2

    try:
        locked = load_locked_scenes(a.splits_dir, list(cfg.locked_scenes) + a.locked_scene)
        water_ids: dict = {}
        negative_weight = 1.0
        if a.smoke:
            cfg.model, cfg.init_weights = "segformer_smoke", "random"
            cfg.image_size, cfg.batch_size, cfg.epochs = 64, 2, 1
            cfg.num_workers, cfg.amp, cfg.class_weight_max_images = 0, False, 50
            smoke_dir = a.output_dir / run_id / "smoke_data"
            tr, va = make_smoke_data(smoke_dir)
            root, manifest_sha, synthetic = smoke_dir, "smoke-inline", True
            tr_scenes, va_scenes = {r.scene_id for r in tr}, {r.scene_id for r in va}
        elif cfg.stage == "pretrain":
            root = a.data_root or ROOT / cfg.data_root
            mpath = a.manifest or ROOT / cfg.manifest
            rows, water_ids = read_seg_manifest(mpath)
            tr = [r for r in rows if r.split == "train"]
            va = [r for r in rows if r.split == "val"]
            manifest_sha, synthetic = file_sha256(mpath), False
            tr_scenes = va_scenes = set()
            from floodcam.seg.classmap import load_classmap

            cm = load_classmap(ROOT / cfg.class_map)
            negative_weight = (
                cfg.negative_weight if cfg.negative_weight is not None
                else cm.negatives.sampling_weight
            )
        else:  # finetune on the scene manifest
            root = a.data_root or ROOT
            mpath = a.manifest or ROOT / cfg.manifest
            spath = a.splits_file or ROOT / cfg.splits_file
            splits = json.loads(Path(spath).read_text(encoding="utf-8"))
            tr_scenes = set(splits["coarse_train_scenes"])
            va_scenes = set(splits["coarse_val_scenes"])
            allrows = rows_from_scene_manifest(mpath, root)
            tr = [r for r in allrows if r.scene_id in tr_scenes]
            va = [r for r in allrows if r.scene_id in va_scenes]
            for r in tr:
                r.split = "train"
            for r in va:
                r.split = "val"
            manifest_sha, synthetic = file_sha256(mpath), any(r.synthetic for r in tr + va)
            negative_weight = cfg.negative_weight or 1.0

        if cfg.stage == "finetune":
            guard_finetune(tr + va, tr_scenes, va_scenes, locked, a.smoke_synthetic)
        else:
            guard_pretrain(tr + va, a.smoke)
        for r in tr:
            r.split = "train"

        metrics = run_training(
            cfg, tr, va, root, water_ids, run_id, a.output_dir, a.results_dir,
            manifest_sha, synthetic, smoke=a.smoke, max_steps=2 if a.smoke else None,
            negative_weight=negative_weight,
            extra_meta={"locked_scenes_checked": sorted(locked)},
        )
    except TrainGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({k: metrics[k] for k in ("run_id", "stage", "best_val_iou", "steps")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
