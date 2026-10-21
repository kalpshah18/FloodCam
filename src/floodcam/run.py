"""CLI entry point: python -m floodcam.run --source <video> --camera <id> --config <yaml>"""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FloodCam: estimate shallow flood depth from fixed-camera video"
    )
    parser.add_argument("--source", required=True, help="Video file path or RTSP URL")
    parser.add_argument("--camera", required=True, help="Camera ID (matches a config YAML)")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--output-dir", default="results", help="Output directory for events")
    parser.parse_args(argv)

    raise NotImplementedError(
        "Pipeline not yet implemented. "
        "Run `make synth-demo` once milestone 4+ is complete."
    )


if __name__ == "__main__":
    sys.exit(main())
