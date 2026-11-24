"""Bounded RealSense inspection; saves aligned RGB-D compatible with scripts/track.py."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[1]


class Args(Tap):
    # Hardware preview: running this entry starts the configured camera, then stops it.
    camera: Path = ROOT / "configs" / "realsense.json"
    frames: int = 30  # Number of unique RGB exposures to retain.
    output: Path = ROOT / "data" / "recordings"


def main(args: Args) -> None:
    from datetime import datetime, timezone
    import json
    import cv2
    import numpy as np
    from mesh_pose.realsense import RealSenseConfig, RealSenseSource

    if args.frames < 2:
        raise ValueError("At least two frames are required to inspect timestamps")
    settings = RealSenseConfig.load(args.camera)
    directory = args.output / ("d415_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    timestamps, frame_metadata = {}, []
    completed = False
    with RealSenseSource(settings) as source:
        (directory / "rgb").mkdir(parents=True, exist_ok=False)
        (directory / "depth").mkdir()
        np.savetxt(directory / "cam_K.txt", source.K)
        try:
            for index in range(args.frames):
                frame = source.read()
                stem = f"{index:06d}"
                raw = np.rint(frame.depth_m / source.depth_scale_m).astype(np.uint16)
                if not cv2.imwrite(str(directory / "rgb" / f"{stem}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("Failed to write RGB image")
                if not cv2.imwrite(str(directory / "depth" / f"{stem}.png"), raw):
                    raise OSError("Failed to write depth image")
                timestamps[stem] = frame.timestamp_s
                frame_metadata.append({"stem": stem, **source.last_frame_metadata,
                    "depth_fraction_0_15_to_2_m": float(np.mean((frame.depth_m >= 0.15) & (frame.depth_m <= 2.0)))})
            completed = True
        finally:
            (directory / "timestamps.json").write_text(json.dumps(timestamps, indent=2), encoding="utf-8")
            report = {**source.metadata, "status": "complete" if completed else "interrupted_or_failed",
                      "frames_saved": len(timestamps), "dropped_color_frames": source.dropped_color_frames,
                      "duplicate_color_frames_skipped": source.duplicate_color_frames, "frames": frame_metadata}
            (directory / "camera.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    first, last = frame_metadata[0], frame_metadata[-1]
    fps = (last["color_frame"] - first["color_frame"]) * 1000 / (last["color_timestamp_ms"] - first["color_timestamp_ms"])
    print(f"Camera released. Saved {len(timestamps)} frames: {directory}")
    print(f"Camera rate: {fps:.2f} fps; last-frame depth coverage within 0.15-2 m: {last['depth_fraction_0_15_to_2_m']:.1%}")


if __name__ == "__main__":
    main(Args().parse_args())
