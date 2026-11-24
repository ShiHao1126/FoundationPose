"""Edit Args defaults, then run: python scripts/track.py. Recorded data only."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[1]


class Args(Tap):
    # Inputs: rectified, aligned RGB-D and calibrated intrinsics.
    sequence: Path = ROOT / "data" / "input"
    mesh: Path = ROOT / "data" / "meshes" / "cover" / "cover.stl"
    mesh_unit: str = "mm"  # cover: user-confirmed mm, bounds 70 x 62 x 70 mm.
    depth_scale_m: float = 0.001  # Raw depth value -> meters.
    fps: float = 30.0  # Used only when timestamps.json is absent; also video playback rate.
    max_frames: int = 0  # 0 means all frames.

    # Shared algorithms, pinned model revisions, outputs.
    config: Path = ROOT / "configs" / "tracker.json"
    models: Path = ROOT / "configs" / "models.json"
    cache: Path = ROOT / "cache"
    output: Path = ROOT / "runs"
    save_video: bool = True


def main(args: Args) -> None:
    import logging
    from mesh_pose.backend import SpatialHubBackend
    from mesh_pose.contracts import Config
    from mesh_pose.sequence import FolderSequence, RunWriter
    from mesh_pose.tracker import Tracker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.mesh.is_file() or not (args.sequence / "cam_K.txt").is_file():
        raise FileNotFoundError("Provide mesh plus sequence/rgb, sequence/depth and sequence/cam_K.txt. "
                                "See README.md; this entry does not connect to a camera.")
    if args.max_frames < 0:
        raise ValueError("max_frames must be >= 0")
    config = Config.load(args.config)
    source = FolderSequence(args.sequence, args.depth_scale_m, args.fps)
    source.frame(0)  # Fail malformed input before loading GPU weights.
    backend = SpatialHubBackend(args.mesh, args.mesh_unit, source.K, config.backend,
                                config.geometry, args.models, args.cache)
    tracker = Tracker(backend, config)
    writer = None
    try:
        writer = RunWriter(args.output, config, {
            "mesh": str(args.mesh.resolve()), "mesh_unit": args.mesh_unit,
            "sequence": str(args.sequence.resolve()), "depth_scale_m": args.depth_scale_m,
            "time_source": source.time_source, "runtime_providers": backend.runtime,
            "T_centered_object": backend.T_centered_object.tolist(),
        }, args.fps, args.save_video)
        for index in range(min(len(source), args.max_frames or len(source))):
            frame = source.frame(index)
            result = tracker.step(frame)
            writer.write(frame, result)
            logging.info("frame=%d state=%s reason=%s elapsed=%.3fs", index, result.state, result.reason, result.elapsed_s)
        writer.mark_complete()
        print(f"Saved run: {writer.directory}")
    finally:
        try:
            tracker.close()
        finally:
            if writer is not None:
                writer.close()


if __name__ == "__main__":
    main(Args().parse_args())
