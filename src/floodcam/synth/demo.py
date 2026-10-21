"""Generate a synthetic dataset for unit tests and dashboard demos.

Writes images, masks, and a manifest to data/synthetic/.
Synthetic data is tagged synthetic=True and MUST NEVER appear in reported results.

Usage:
    python -m floodcam.synth.demo [--out-dir data/synthetic] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent.parent.parent  # project root

# Try to import from installed package; fall back to direct import for dev
try:
    from floodcam.io.manifest import ManifestRow, write_jsonl
    from floodcam.synth.renderer import SceneConfig, SynthRenderer
except ImportError:
    # Running as script before install
    sys.path.insert(0, str(ROOT / "src"))
    from floodcam.io.manifest import ManifestRow, write_jsonl
    from floodcam.synth.renderer import SceneConfig, SynthRenderer


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

_NORMAL_SCENE_PARAMS = [
    # (camera_height_cm, tilt_deg, focal_px, pole_dist_cm, lighting_tag)
    (120.0, 20.0, 600.0, 200.0, "synthetic_indoor_diffuse"),
    (130.0, 18.0, 580.0, 190.0, "synthetic_indoor_directional"),
    (110.0, 22.0, 620.0, 210.0, "synthetic_outdoor_overcast"),
    (125.0, 25.0, 650.0, 180.0, "synthetic_outdoor_bright"),
    (115.0, 15.0, 560.0, 220.0, "synthetic_indoor_low"),
    (135.0, 28.0, 700.0, 175.0, "synthetic_outdoor_harsh"),
    (100.0, 17.0, 540.0, 230.0, "synthetic_indoor_side"),
    (140.0, 23.0, 640.0, 195.0, "synthetic_outdoor_cloudy"),
]

_NEGATIVE_SCENE_PARAMS = [
    (120.0, 20.0, 600.0, 200.0, "reflection", "synthetic_reflection"),
    (120.0, 20.0, 600.0, 200.0, "wet_road", "synthetic_wet_road"),
]

_DEPTHS_CM = [0.0, 2.0, 3.0, 5.0, 10.0, 20.0]
_FRAMES_PER_DEPTH = 2  # frames per depth level per scene
_FRAMES_PER_NEG_SCENE = 5  # frames per negative scene


def _make_scene_id(idx: int, neg: bool = False, variant: str = "") -> str:
    if neg:
        return f"synth_neg_{variant}_{idx:02d}"
    return f"synth_scene_{idx:02d}"


def generate(out_dir: Path, seed: int = 42) -> Path:
    """Generate the full synthetic dataset. Returns path to manifest JSONL."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    rows: list[ManifestRow] = []
    frame_counter = 0

    # ---- Normal scenes ---------------------------------------------------
    for scene_idx, (h, tilt, f, dist, lighting) in enumerate(_NORMAL_SCENE_PARAMS):
        scene_id = _make_scene_id(scene_idx)
        scene_seed = seed * 1000 + scene_idx

        cfg = SceneConfig(
            camera_height_cm=h,
            camera_tilt_deg=tilt,
            focal_length_px=f,
            pole_distance_cm=dist,
        )
        renderer = SynthRenderer(cfg, seed=scene_seed)

        for depth_cm in _DEPTHS_CM:
            for fi in range(_FRAMES_PER_DEPTH):
                frame_id = f"frame_{frame_counter:05d}"
                img_path = img_dir / f"{frame_id}.png"
                msk_path = mask_dir / f"{frame_id}.png"

                frame = renderer.render_frame(depth_cm, frame_idx=fi, variant="normal")

                Image.fromarray(frame.image).save(img_path)
                Image.fromarray(frame.mask).save(msk_path)

                rows.append(
                    ManifestRow(
                        frame_id=frame_id,
                        scene_id=scene_id,
                        session_id=scene_id,
                        camera_id=f"synth_cam_{scene_idx:02d}",
                        image_path=str(img_path.relative_to(ROOT)),
                        mask_path=str(msk_path.relative_to(ROOT)),
                        depth_cm=frame.depth_cm,
                        lighting=lighting,
                        is_negative=False,
                        synthetic=True,
                        split="",
                    )
                )
                frame_counter += 1

    # ---- Hard-negative scenes --------------------------------------------
    for neg_idx, (h, tilt, f, dist, variant, lighting) in enumerate(_NEGATIVE_SCENE_PARAMS):
        scene_id = _make_scene_id(neg_idx, neg=True, variant=variant)
        scene_seed = seed * 1000 + len(_NORMAL_SCENE_PARAMS) + neg_idx

        cfg = SceneConfig(
            camera_height_cm=h,
            camera_tilt_deg=tilt,
            focal_length_px=f,
            pole_distance_cm=dist,
        )
        renderer = SynthRenderer(cfg, seed=scene_seed)

        for fi in range(_FRAMES_PER_NEG_SCENE):
            frame_id = f"frame_{frame_counter:05d}"
            img_path = img_dir / f"{frame_id}.png"
            msk_path = mask_dir / f"{frame_id}.png"

            # Negatives rendered with depth 0 and their specific variant
            frame = renderer.render_frame(0.0, frame_idx=fi, variant=variant)

            Image.fromarray(frame.image).save(img_path)
            Image.fromarray(frame.mask).save(msk_path)

            rows.append(
                ManifestRow(
                    frame_id=frame_id,
                    scene_id=scene_id,
                    session_id=scene_id,
                    camera_id=f"synth_neg_cam_{neg_idx:02d}",
                    image_path=str(img_path.relative_to(ROOT)),
                    mask_path=str(msk_path.relative_to(ROOT)),
                    depth_cm=0.0,
                    lighting=lighting,
                    is_negative=True,
                    synthetic=True,
                    split="",
                )
            )
            frame_counter += 1

    manifest_path = out_dir / "manifest.jsonl"
    count = write_jsonl(rows, manifest_path)
    print(
        f"Wrote {count} frames from {len(_NORMAL_SCENE_PARAMS)} normal + "
        f"{len(_NEGATIVE_SCENE_PARAMS)} negative scenes to {out_dir}"
    )
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
