"""Calibration annotation tool.

Two modes:
  Interactive — OpenCV window: left-click reference marks, right-click ROI
                vertices, press 's' to save.
  Headless    — read points from a JSON file (--points-json), write YAML.

CLI usage:
    python -m floodcam.calib.annotate \\
        --image path/to/frame.jpg \\
        --camera-id cam_01 \\
        --output configs/cam_01.yaml \\
        [--mode linear|homography]

    python -m floodcam.calib.annotate \\
        --points-json calib_pts.json \\
        --camera-id cam_01 \\
        --output configs/cam_01.yaml \\
        [--mode linear|homography]

Headless JSON format (linear mode — 2-element marks):
    {
        "reference_marks": [[pixel_row, height_cm], ...],
        "roi_polygon":      [[x, y], ...]   // optional
    }

Headless JSON format (homography mode — 4-element marks):
    {
        "reference_marks": [[image_u, image_v, plane_x_cm, height_cm], ...],
        "roi_polygon":      [[x, y], ...]   // optional
    }

The mode is inferred from the mark length unless --mode is specified explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from floodcam.calib.homography import HomographyCalib
from floodcam.calib.linear import CalibError, LinearCalib
from floodcam.config import ROI, CameraConfig

# ------------------------------------------------------------------ #
# Public headless entry point (also used by tests)
# ------------------------------------------------------------------ #


def annotate_headless(
    points_path: Path,
    camera_id: str,
    output_path: Path,
    mode: Optional[str] = None,
) -> CameraConfig:
    """Fit calibration from a JSON points file and save to a camera YAML.

    Args:
        points_path: JSON file with reference_marks and optional roi_polygon.
        camera_id:   Camera ID written into the YAML.
        output_path: Path for the output YAML (created or overwritten).
        mode:        "linear" | "homography" | None (auto-detect from mark length).

    Returns:
        The saved CameraConfig.

    Raises:
        CalibError: invalid points, wrong format, or calibration failure.
        FileNotFoundError: points_path does not exist.
    """
    pts_data = json.loads(points_path.read_text())
    marks = pts_data.get("reference_marks")
    if marks is None or len(marks) == 0:
        raise CalibError("JSON must contain a non-empty 'reference_marks' list.")

    # Infer mode from mark length if not given
    mark_len = len(marks[0])
    if mode is None:
        if mark_len == 2:
            mode = "linear"
        elif mark_len == 4:
            mode = "homography"
        else:
            raise CalibError(
                f"Cannot infer calibration mode from mark length {mark_len}. "
                "Expected 2 (linear) or 4 (homography)."
            )
    elif mode == "linear" and mark_len not in (2,):
        raise CalibError(
            f"linear mode expects 2-element marks [row, height_cm]; got {mark_len}."
        )
    elif mode == "homography" and mark_len not in (4,):
        raise CalibError(
            f"homography mode expects 4-element marks [u, v, x_cm, h_cm]; got {mark_len}."
        )

    # Load or create config
    cfg: CameraConfig
    if output_path.exists():
        cfg = CameraConfig.load(output_path)
        cfg.camera_id = camera_id
    else:
        cfg = CameraConfig(camera_id=camera_id)

    cfg.calibration.mode = mode
    cfg.calibration.reference_points = [list(map(float, m)) for m in marks]

    # Fit
    if mode == "linear":
        _fit_linear(marks, cfg)
    else:
        _fit_homography(marks, cfg)

    # Optional ROI polygon
    polygon = pts_data.get("roi_polygon")
    if polygon:
        _update_roi_from_polygon(polygon, cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(output_path)
    return cfg


def _fit_linear(marks: list, cfg: CameraConfig) -> None:
    points = [(float(m[0]), float(m[1])) for m in marks]
    lc = LinearCalib().fit(points)
    updates = lc.to_config_updates()
    cfg.calibration.scale_cm_per_px = updates["scale_cm_per_px"]
    cfg.calibration.intercept_cm = updates["intercept_cm"]
    cfg.calibration.sigma_s = updates["sigma_s"]


def _fit_homography(marks: list, cfg: CameraConfig) -> None:
    points = [
        (float(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in marks
    ]
    hc = HomographyCalib().fit(points)
    updates = hc.to_config_updates()
    cfg.calibration.homography_matrix = updates["homography_matrix"]


def _update_roi_from_polygon(polygon: list, cfg: CameraConfig) -> None:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    cfg.roi = ROI(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


# ------------------------------------------------------------------ #
# Interactive mode (OpenCV)
# ------------------------------------------------------------------ #


def annotate_interactive(
    image_path: Path,
    camera_id: str,
    output_path: Path,
    mode: str = "linear",
) -> None:
    """Interactive OpenCV annotation window.

    Left-click  : add a reference mark (terminal will prompt for height in cm)
    Right-click : add an ROI vertex
    u key       : undo last left-click mark
    s key       : fit calibration and save
    q key       : quit without saving
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for interactive mode.") from exc

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ref_marks: list[tuple[float, float]] = []   # (pixel_row, height_cm)
    roi_poly: list[tuple[int, int]] = []

    win = "FloodCam Calibration — L:add mark  R:roi vertex  u:undo  s:save  q:quit"
    cv2.namedWindow(win)

    def _redraw():
        disp = img.copy()
        for r, h in ref_marks:
            cv2.circle(disp, (img.shape[1] // 2, int(r)), 5, (0, 255, 0), -1)
            cv2.putText(disp, f"{h:.1f}cm", (img.shape[1] // 2 + 8, int(r)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for i, pt in enumerate(roi_poly):
            cv2.circle(disp, pt, 4, (0, 128, 255), -1)
            if i > 0:
                cv2.line(disp, roi_poly[i - 1], pt, (0, 128, 255), 1)
        cv2.imshow(win, disp)

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_str = input(f"  Height at row {y} (cm): ").strip()
            try:
                h_val = float(h_str)
            except ValueError:
                print("  Invalid height, skipping.")
                return
            ref_marks.append((float(y), h_val))
            print(f"  Added mark: row={y}, height={h_val} cm  (total {len(ref_marks)})")
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            roi_poly.append((x, y))
            print(f"  ROI vertex ({x}, {y}) — total {len(roi_poly)}")
            _redraw()

    cv2.setMouseCallback(win, _mouse)
    _redraw()

    print("Instructions: left-click reference marks (height prompted), "
          "right-click ROI vertices, s=save, q=quit, u=undo.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return
        elif key == ord("u") and ref_marks:
            removed = ref_marks.pop()
            print(f"  Undone mark at row {removed[0]:.0f}")
            _redraw()
        elif key == ord("s"):
            if len(ref_marks) < 2:
                print("  Need at least 2 reference marks before saving.")
                continue
            break

    cv2.destroyAllWindows()

    # Write temporary JSON and reuse annotate_headless
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".json"))
    payload: dict = {"reference_marks": [[r, h] for r, h in ref_marks]}
    if roi_poly:
        payload["roi_polygon"] = [list(p) for p in roi_poly]
    tmp.write_text(json.dumps(payload))
    try:
        cfg = annotate_headless(tmp, camera_id, output_path, mode=mode)
        print(f"Saved calibration to {output_path}")
        print(f"  scale_cm_per_px = {cfg.calibration.scale_cm_per_px:.5f}")
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FloodCam calibration annotation tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--image", help="Frame image (required for interactive mode)")
    p.add_argument("--points-json", help="JSON file for headless mode")
    p.add_argument("--camera-id", default="cam_default", help="Camera ID for the output YAML")
    p.add_argument("--output", required=True, help="Output camera YAML path")
    p.add_argument(
        "--mode",
        choices=["linear", "homography"],
        default=None,
        help="Calibration mode (auto-detected from JSON mark length if omitted)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = Path(args.output)

    if args.points_json:
        pts_path = Path(args.points_json)
        if not pts_path.exists():
            print(f"Error: points JSON not found: {pts_path}", file=sys.stderr)
            return 1
        try:
            cfg = annotate_headless(pts_path, args.camera_id, output_path, mode=args.mode)
            print(f"Saved: {output_path}")
            if cfg.calibration.scale_cm_per_px is not None:
                print(f"  scale_cm_per_px = {cfg.calibration.scale_cm_per_px:.5f} cm/px")
            if cfg.calibration.sigma_s is not None:
                print(f"  sigma_s         = {cfg.calibration.sigma_s:.5f} cm/px")
        except (CalibError, ValueError) as exc:
            print(f"Calibration error: {exc}", file=sys.stderr)
            return 1
    elif args.image:
        try:
            annotate_interactive(
                Path(args.image),
                args.camera_id,
                output_path,
                mode=args.mode or "linear",
            )
        except (CalibError, RuntimeError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        print("Provide --points-json (headless) or --image (interactive).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
