"""Milestone 8 tests: class map, adapters, leakage-safe statistics, training guards,
smoke training, SegFormerBackend protocol compliance, optional-torch import.

Tests marked `torch` skip when the [ml] extra is missing. Nothing here downloads anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from floodcam.seg.classmap import ClassMapError, load_classmap, resolve_water_ids
from floodcam.seg.datasets import (
    SegDataset,
    SegRow,
    adapt_dataset,
    adapt_negatives,
    binarize_mask,
    build_public_manifest,
    compute_class_weights,
    read_seg_manifest,
    write_public_splits,
    write_seg_manifest,
)
from floodcam.seg.train import (
    TrainGuardError,
    guard_finetune,
    guard_pretrain,
    load_locked_scenes,
)
from floodcam.seg.train import (
    main as train_main,
)

SRC = Path(__file__).parent.parent / "src"

# ---------------------------------------------------------------------------
# Tiny fixture datasets (ids deliberately arbitrary: the label files are the truth)
# ---------------------------------------------------------------------------


def _img(path: Path, color=(120, 90, 60), size=(24, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), color, np.uint8))


def _mask(path: Path, values: dict[tuple[int, int], int], size=(24, 32)) -> None:
    """values: {(row0,row1): id} horizontal bands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros(size, np.uint8)
    for (r0, r1), v in values.items():
        m[r0:r1] = v
    cv2.imwrite(str(path), m)


@pytest.fixture
def public_root(tmp_path):
    root = tmp_path / "public"
    # ADE-style: csv with Name column; row order gives id (offset 1)
    ade = root / "ade20k"
    (ade).mkdir(parents=True)
    (ade / "objectInfo150.csv").write_text(
        "Idx,Ratio,Train,Val,Name\n1,0.1,1,1,wall\n2,0.1,1,1,sky\n3,0.1,1,1,sea;ocean\n"
        "4,0.1,1,1,river\n", encoding="utf-8")
    for split, n in (("training", 2), ("validation", 1)):
        for i in range(n):
            _img(ade / "images" / split / f"a_{split}_{i}.jpg")
            _mask(ade / "annotations" / split / f"a_{split}_{i}.png", {(0, 8): 3, (8, 12): 1})
    # COCO-style: "<id>: <name>" label file
    coco = root / "cocostuff"
    coco.mkdir()
    (coco / "labels.txt").write_text("0: unlabeled\n92: river\n93: sea\n94: water-other\n",
                                     encoding="utf-8")
    for split, n in (("train2017", 1), ("val2017", 1)):
        for i in range(n):
            _img(coco / "images" / split / f"c_{split}_{i}.jpg")
            _mask(coco / "stuffthingmaps" / split / f"c_{split}_{i}.png", {(0, 6): 92, (6, 9): 0})
    # generic binary export
    gen = root / "generic_flood"
    for split, n in (("train", 1), ("valid", 1)):
        for i in range(n):
            _img(gen / "images" / split / f"g_{split}_{i}.jpg")
            _mask(gen / "masks" / split / f"g_{split}_{i}.png", {(0, 10): 255})
    # negatives: two sequences, one with a mask, one without
    neg = root / "negatives"
    for seq in ("rain_seq", "puddle_seq"):
        for i in range(3):
            _img(neg / "images" / seq / f"f{i}.jpg")
    _mask(neg / "masks" / "puddle_seq" / "f0.png", {(20, 22): 255})
    return root


def _classmap_yaml(tmp_path, public_root, ade_names=("water_missing",), floodnet=False) -> Path:
    cfg = {
        "version": 1,
        "datasets": {
            "ade20k": {
                "enabled": True, "root": "ade20k", "label_file": "objectInfo150.csv",
                "label_format": "ade_objectinfo_csv", "water_class_names": list(ade_names),
                "layout": {"images": "images/{split}", "masks": "annotations/{split}",
                           "image_ext": [".jpg"], "mask_ext": ".png"},
                "split_dirs": {"train": "training", "val": "validation"},
            },
            "cocostuff": {
                "enabled": True, "root": "cocostuff", "label_file": "labels.txt",
                "label_format": "id_name_txt", "water_class_names": ["river", "sea"],
                "layout": {"images": "images/{split}", "masks": "stuffthingmaps/{split}",
                           "image_ext": [".jpg"], "mask_ext": ".png"},
                "split_dirs": {"train": "train2017", "val": "val2017"},
            },
            "floodnet": {
                "enabled": floodnet, "root": "floodnet", "label_file": "classes.txt",
                "label_format": "id_name_txt", "water_class_names": ["Water"],
                "layout": {"images": "{split}/image", "masks": "{split}/label"},
                "split_dirs": {"train": "train", "val": "val"},
            },
            "generic_flood": {
                "enabled": True, "root": "generic_flood", "label_file": None,
                "label_format": "binary",
                "layout": {"images": "images/{split}", "masks": "masks/{split}",
                           "image_ext": [".jpg"], "mask_ext": ".png"},
                "split_dirs": {"train": "train", "val": "valid"},
            },
        },
        "negatives": {"enabled": True, "root": "negatives", "sampling_weight": 3.0,
                      "val_fraction": 0.5, "image_ext": [".jpg"]},
    }
    p = tmp_path / "classmap.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------


def test_repo_classmap_loads_and_floodnet_off_by_default():
    cm = load_classmap(Path(__file__).parent.parent / "configs" / "seg_classmap.yaml")
    assert cm.datasets["floodnet"].enabled is False
    assert cm.datasets["floodnet"].domain_gap_note
    assert all(not d.license_verified for d in cm.datasets.values())


def test_classmap_fails_on_missing_class_name(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("water_missing",)))
    with pytest.raises(ClassMapError, match="water_missing"):
        resolve_water_ids(cm.datasets["ade20k"], public_root / "ade20k")


def test_classmap_fails_on_missing_class_id(tmp_path, public_root):
    p = _classmap_yaml(tmp_path, public_root, ade_names=("sea",))
    raw = yaml.safe_load(p.read_text())
    raw["datasets"]["ade20k"]["water_class_ids"] = [999]
    p.write_text(yaml.safe_dump(raw))
    cm = load_classmap(p)
    with pytest.raises(ClassMapError, match="999"):
        resolve_water_ids(cm.datasets["ade20k"], public_root / "ade20k")


def test_classmap_fails_when_label_file_missing(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    (public_root / "ade20k" / "objectInfo150.csv").unlink()
    with pytest.raises(ClassMapError, match="label file not found"):
        resolve_water_ids(cm.datasets["ade20k"], public_root / "ade20k")


def test_classmap_resolves_synonym_names(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("ocean", "river")))
    assert resolve_water_ids(cm.datasets["ade20k"], public_root / "ade20k") == [3, 4]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_adapters_produce_binary_masks(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    rows, water = build_public_manifest(cm, public_root)
    assert water == {"ade20k": [3], "cocostuff": [92, 93], "generic_flood": None}
    for src in ("ade20k", "cocostuff", "generic_flood"):
        r = next(r for r in rows if r.source == src and r.split == "train")
        ds = SegDataset([r], public_root, water, image_size=16)
        x, y = ds[0]
        assert x.shape == (3, 16, 16) and x.dtype == np.float32
        assert set(np.unique(y)) <= {0, 1} and y.sum() > 0
    # ADE: id 1 (wall) must NOT be water; only the id-3 band is
    r = next(r for r in rows if r.source == "ade20k")
    _, mask = SegDataset([r], public_root, water, 32).load_raw(0)
    assert mask[:8].all() and not mask[8:].any()


def test_official_splits_preserved(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    rows, _ = build_public_manifest(cm, public_root)
    ade = [r for r in rows if r.source == "ade20k"]
    assert sum(r.split == "train" for r in ade) == 2 and sum(r.split == "val" for r in ade) == 1


def test_negatives_empty_and_partial_masks(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    rows = adapt_negatives(cm, public_root)
    assert len(rows) == 6 and all(r.is_negative for r in rows)
    ds = SegDataset(rows, public_root, {}, 16)
    sums = {r.image_id: ds.load_raw(i)[1].sum() for i, r in enumerate(rows)}
    assert sums["negatives/puddle_seq/f0"] > 0  # partial mask
    assert sums["negatives/puddle_seq/f1"] == 0 and sums["negatives/rain_seq/f0"] == 0
    # sequence-level split: frames of one sequence never straddle train/val
    for seq in ("rain_seq", "puddle_seq"):
        assert len({r.split for r in rows if f"/{seq}/" in r.image_id}) == 1


def test_adapter_fails_loudly_on_missing_mask(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    next((public_root / "ade20k" / "annotations" / "training").glob("*.png")).unlink()
    with pytest.raises(Exception, match="no mask"):
        adapt_dataset(cm.datasets["ade20k"], public_root)


def test_binarize_mask_rules():
    m = np.array([[0, 1, 3], [3, 4, 0]], np.uint8)
    assert binarize_mask(m, [3]).tolist() == [[0, 0, 1], [1, 0, 0]]
    assert binarize_mask(m, None).tolist() == [[0, 1, 1], [1, 1, 0]]


def test_splits_file_ids_only_and_manifest_roundtrip(tmp_path, public_root):
    cm = load_classmap(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    rows, water = build_public_manifest(cm, public_root)
    sha = write_seg_manifest(rows, water, tmp_path / "m.jsonl")
    back, water2 = read_seg_manifest(tmp_path / "m.jsonl")
    assert [r.image_id for r in back] == [r.image_id for r in rows] and water2 == water
    write_public_splits(rows, sha, tmp_path / "splits.json")
    doc = json.loads((tmp_path / "splits.json").read_text())
    assert doc["manifest_sha256"] == sha
    flat = json.dumps(doc["sources"])
    assert ".jpg" not in flat and "/" in flat  # ids like "ade20k/a_training_0", no file paths
    ids = [i for s in doc["sources"].values() for split in s.values() for i in split]
    assert len(ids) == len(set(ids)) == len(rows)


def test_prepare_script_verify_and_fail_loudly(tmp_path, public_root):
    script = Path(__file__).parent.parent / "scripts" / "prepare_public_data.py"
    out = tmp_path / "splits.json"

    def run(cmap):
        return subprocess.run(
            [sys.executable, str(script), "--verify", "--class-map", str(cmap),
             "--data-root", str(public_root), "--splits-out", str(out)],
            capture_output=True, text=True)

    bad = run(_classmap_yaml(tmp_path, public_root, ade_names=("water_missing",)))
    assert bad.returncode == 1 and "water_missing" in bad.stderr and not out.exists()
    good = run(_classmap_yaml(tmp_path, public_root, ade_names=("sea",)))
    assert good.returncode == 0, good.stderr
    md = (public_root / "DATASETS.md").read_text()
    assert "ade20k" in md and "licence not marked verified" in md
    assert json.loads(out.read_text())["type"] == "public_seg"
    # default invocation downloads / writes nothing
    info = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert info.returncode == 0 and "downloaded" in info.stdout


# ---------------------------------------------------------------------------
# Leakage: statistics from the training split only
# ---------------------------------------------------------------------------


def _stat_rows(tmp_path, val_water_rows: int):
    rows = []
    for split, band in (("train", 4), ("val", val_water_rows)):
        for i in range(2):
            _img(tmp_path / f"{split}{i}.png")
            _mask(tmp_path / f"{split}{i}_m.png", {(0, band): 255} if band else {})
            rows.append(SegRow(f"{split}{i}", f"{split}{i}m", "generic", split, False,
                               f"{split}{i}.png", f"{split}{i}_m.png"))
    return rows


def test_class_weights_use_train_rows_only(tmp_path):
    rows = _stat_rows(tmp_path, val_water_rows=20)
    train = [r for r in rows if r.split == "train"]
    w1 = compute_class_weights(train, tmp_path, {})
    # changing validation masks must not change the weights
    rows2 = _stat_rows(tmp_path, val_water_rows=0)
    w2 = compute_class_weights([r for r in rows2 if r.split == "train"], tmp_path, {})
    assert w1 == w2 and w1[1] > w1[0]  # water is the rarer class -> larger weight
    with pytest.raises(ValueError, match="train rows only"):
        compute_class_weights(rows, tmp_path, {})


# ---------------------------------------------------------------------------
# Augmentation (no torch)
# ---------------------------------------------------------------------------


def test_augment_deterministic_and_mask_stays_binary():
    from floodcam.seg.augment import AugConfig, augment

    rng_img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    m = np.zeros((48, 64), np.uint8)
    m[24:] = 1
    cfg = AugConfig(hflip_p=1, crop_p=1, jitter_p=1, blur_p=1, noise_p=1, jpeg_p=1, rain_p=1)
    a = augment(rng_img, m, np.random.default_rng(7), cfg)
    b = augment(rng_img, m, np.random.default_rng(7), cfg)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert a[0].shape == rng_img.shape and a[0].dtype == np.uint8
    assert set(np.unique(a[1])) <= {0, 1}


# ---------------------------------------------------------------------------
# Guards (no torch needed)
# ---------------------------------------------------------------------------


def _row(scene, synthetic=False):
    return SegRow(f"{scene}_x", f"{scene}_m", "self_labeled", "train", False, "a.png", "b.png",
                  scene_id=scene, synthetic=synthetic)


def test_guard_refuses_locked_scene():
    with pytest.raises(TrainGuardError, match="locked"):
        guard_finetune([_row("s1"), _row("s9")], ["s1"], ["s9"], {"s9"}, allow_synthetic=False)


def test_guard_refuses_synthetic_without_flag():
    with pytest.raises(TrainGuardError, match="synthetic"):
        guard_finetune([_row("s1", True)], ["s1"], ["s2"], set(), allow_synthetic=False)
    guard_finetune([_row("s1", True)], ["s1"], ["s2"], set(), allow_synthetic=True)


def test_guard_refuses_scene_in_train_and_val():
    with pytest.raises(TrainGuardError, match="both"):
        guard_finetune([_row("s1")], ["s1"], ["s1"], set(), allow_synthetic=False)


def test_guard_pretrain_refuses_synthetic_outside_smoke():
    with pytest.raises(TrainGuardError):
        guard_pretrain([_row("s1", True)], smoke=False)
    guard_pretrain([_row("s1", True)], smoke=True)


def test_locked_scenes_read_from_splits_dir(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"locked_test_scene": "lockedA"}))
    (tmp_path / "b.json").write_text(json.dumps({"locked_test_scene": ["lockedB"]}))
    (tmp_path / "c.json").write_text(json.dumps({"no": "lock"}))
    assert load_locked_scenes(tmp_path, ["x"]) == {"lockedA", "lockedB", "x"}


def test_cli_finetune_smoke_refused_without_smoke_synthetic(tmp_path):
    rc = train_main(["--smoke", "--stage", "finetune", "--output-dir", str(tmp_path / "ck"),
                     "--results-dir", str(tmp_path / "res"), "--splits-dir", str(tmp_path)])
    assert rc == 3 and not (tmp_path / "res" / "seg-finetune-smoke-001").exists()


def test_cli_finetune_refuses_locked_scene(tmp_path):
    from floodcam.io.manifest import ManifestRow, write_jsonl

    rows = [ManifestRow(f"f{i}", sc, sc, "cam", f"{sc}.png", f"{sc}_m.png", 5.0, "x", False,
                        False) for i, sc in enumerate(["trainA", "valB", "lockedC"])]
    write_jsonl(rows, tmp_path / "manifest.jsonl")
    splits = {"coarse_train_scenes": ["trainA", "lockedC"], "coarse_val_scenes": ["valB"],
              "locked_test_scene": "lockedC"}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    rc = train_main(["--stage", "finetune", "--manifest", str(tmp_path / "manifest.jsonl"),
                     "--splits-file", str(tmp_path / "splits.json"),
                     "--splits-dir", str(tmp_path), "--results-dir", str(tmp_path / "res"),
                     "--output-dir", str(tmp_path / "ck")])
    assert rc == 3 and not (tmp_path / "res").exists()


# ---------------------------------------------------------------------------
# Torch-dependent: smoke training, backend, runner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    td = tmp_path_factory.mktemp("smoke")
    rc = train_main(["--smoke", "--output-dir", str(td / "ck"), "--results-dir", str(td / "res"),
                     "--splits-dir", str(td)])
    assert rc == 0
    run_id = "seg-pretrain-smoke-001"
    return td / "res" / run_id, td / "ck" / run_id


@pytest.mark.torch
def test_smoke_training_writes_outputs(smoke_run):
    res, ck = smoke_run
    for name in ("metrics.json", "curves.csv", "meta.json"):
        assert (res / name).is_file()
    metrics = json.loads((res / "metrics.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    assert metrics["steps"] == 2 and metrics["synthetic"] is True and metrics["smoke"] is True
    assert metrics["class_weights_train_only"] and meta["seed"] == 42
    for key in ("git_commit", "torch", "transformers", "hardware", "dataset_manifest_sha256",
                "init_weights", "config"):
        assert key in meta
    assert meta["locked_scene_touched"] is False
    assert (ck / "best" / "floodcam_meta.json").is_file()
    assert (ck / "last" / "floodcam_meta.json").is_file()


@pytest.mark.torch
def test_smoke_checkpoint_loadable(smoke_run):
    from transformers import SegformerForSemanticSegmentation

    _, ck = smoke_run
    m = SegformerForSemanticSegmentation.from_pretrained(ck / "best")
    assert m.config.num_labels == 2


@pytest.mark.torch
def test_backend_protocol_and_shape(smoke_run):
    from floodcam.config import ExperimentConfig
    from floodcam.seg.segformer import SegFormerBackend

    _, ck = smoke_run
    b = SegFormerBackend(ExperimentConfig(seg_prob_threshold=0.5), ck / "best", device="cpu")
    assert isinstance(b.model_version, str) and b.model_version.startswith("segformer_smoke-")
    assert len(b.model_version.split("-")[-1]) == 8
    frame = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    mask = b.predict_mask(frame)
    assert mask.shape == (60, 80) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    hi = SegFormerBackend(ExperimentConfig(seg_prob_threshold=1.01), ck / "best", device="cpu")
    assert hi.predict_mask(frame).sum() == 0  # threshold is honoured


@pytest.mark.torch
def test_backend_env_var_and_missing_ckpt(smoke_run, monkeypatch):
    from floodcam.seg.segformer import CKPT_ENV, SegFormerBackend

    _, ck = smoke_run
    monkeypatch.setenv(CKPT_ENV, str(ck))  # run dir containing best/
    assert SegFormerBackend(device="cpu").model_version.startswith("segformer_smoke-")
    monkeypatch.delenv(CKPT_ENV)
    with pytest.raises(FileNotFoundError):
        SegFormerBackend(device="cpu")


@pytest.mark.torch
def test_backend_runs_through_runner(smoke_run, tmp_path, monkeypatch):
    """Protocol compliance only: an untrained smoke model makes no accuracy claim."""
    from floodcam.config import ExperimentConfig
    from floodcam.depth.estimate import make_camera_config_for_renderer
    from floodcam.run import process_timelapse
    from floodcam.seg.segformer import SegFormerBackend
    from floodcam.store.db import EventStore
    from floodcam.synth.renderer import SceneConfig, SynthRenderer
    from floodcam.synth.timelapse import DEMO_SCENE_ID, render_timelapse
    from floodcam.synth.timeseries import steady_rise

    monkeypatch.chdir(tmp_path)  # runner writes results/runs/<id>/snapshots relative to cwd
    _, ck = smoke_run
    rend = SynthRenderer(SceneConfig(110.0, 22.0, 620.0, pole_distance_cm=210.0), seed=42002)
    cam = make_camera_config_for_renderer(rend)
    cam.camera_id = "seg_cam"
    exp = ExperimentConfig(hsv_lower=[90, 30, 30], hsv_upper=[120, 255, 255])
    tl = render_timelapse(steady_rise(0.0, 10.0, 12.0, 1.0, seed=1), tmp_path / "tl",
                          scene_id=DEMO_SCENE_ID)
    backend = SegFormerBackend(exp, ck / "best", device="cpu")
    db = tmp_path / "e.db"
    summary = process_timelapse(tl, cam, exp, db, "segrun", backend, synthetic=True,
                                snapshot_every_s=5.0)
    assert summary["event_count"] > 0
    store = EventStore(db)
    ev = store.events_for_run("segrun")
    store.close()
    assert ev and all(e["model_version"] == backend.model_version for e in ev)
    assert all(e["synthetic"] == 1 for e in ev)


# ---------------------------------------------------------------------------
# Importing floodcam without torch
# ---------------------------------------------------------------------------


def test_import_floodcam_without_torch():
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, r"{SRC}")
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "transformers", "timm", "accelerate"):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        import floodcam, floodcam.seg, floodcam.seg.segformer, floodcam.seg.datasets
        import floodcam.seg.classmap, floodcam.seg.augment, floodcam.seg.train
        import floodcam.run, floodcam.depth.estimate
        from floodcam.seg.segformer import SegFormerBackend
        try:
            SegFormerBackend(ckpt_dir="x")
        except ImportError as e:
            assert "pip install" in str(e), e
        else:
            raise SystemExit("expected ImportError")
        assert "torch" not in sys.modules
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
