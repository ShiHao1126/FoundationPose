"""Edit Args defaults and run python scripts/track_live.py to open the live camera."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[1]


class Args(Tap):
    # Real hardware. The visible preview opens the configured camera when enabled.
    connect_camera: bool = True
    camera: Path = ROOT / "configs" / "realsense.json"
    mesh: Path = ROOT / "data" / "meshes" / "cover" / "cover.stl"
    mesh_unit: str = "mm"  # User-confirmed 70 x 62 x 70 mm cover.

    # Shared algorithms and fixed model revisions.
    config: Path = ROOT / "configs" / "tracker.json"
    models: Path = ROOT / "configs" / "models.json"
    cache: Path = ROOT / "cache"

    # Display and diagnostics. Pose age uses host monotonic arrival time, seconds.
    max_pose_age_s: float = 0.5
    duration_s: float = 0.0  # 0: until Stop, Esc, or window close.
    output: Path = ROOT / "runs" / "live"
    save_video: bool = False  # Enable for occlusion diagnostics; normal preview saves poses and transitions.
    video_fps: float = 5.0  # Diagnostic playback only; timing.jsonl is authoritative.


def main(args: Args):
    import logging
    import signal
    from mesh_pose.backend import SpatialHubBackend
    from mesh_pose.contracts import Config
    from mesh_pose.live import LiveSession
    from mesh_pose.realsense import RealSenseConfig, RealSenseSource
    from mesh_pose.sequence import RunWriter
    from mesh_pose.viewer import LiveViewer

    if not args.connect_camera:
        print("connect_camera=False; camera not opened")
        return
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)
    if args.max_pose_age_s <= 0 or args.duration_s < 0 or args.video_fps <= 0:
        raise ValueError("Pose age/video fps must be positive; duration must be nonnegative")
    config = Config.load(args.config)
    if config.tracking.confirmation_frames < 2:
        raise ValueError("Live input requires at least two confirmation frames")
    camera = RealSenseConfig.load(args.camera)
    writer = RunWriter(args.output, config, {
        "source": "live_realsense", "mesh": str(args.mesh.resolve()), "mesh_unit": args.mesh_unit,
        "max_pose_age_s": args.max_pose_age_s, "video_playback_fps": args.video_fps,
        "time_source": "camera SDK timestamps; host monotonic timing in timing.jsonl",
    }, args.video_fps, args.save_video)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(writer.directory / "runtime.log", encoding="utf-8"), logging.StreamHandler()])
    session = LiveSession(RealSenseSource(camera), lambda K: SpatialHubBackend(
        args.mesh, args.mesh_unit, K, config.backend, config.geometry, args.models, args.cache), config, writer)
    print(f"Live run: {writer.directory}", flush=True)
    signal.signal(signal.SIGINT, lambda signum, frame: session.request_stop())
    try:
        viewer = LiveViewer(session, args.max_pose_age_s, args.duration_s, args.save_video)
        session.start()
        viewer.run()
    except Exception:
        session.fail("display/startup")
        raise
    finally:
        session.close()
    error = session.snapshot()["error"]
    if error:
        raise RuntimeError(error)
    print(f"Camera released. Saved run: {writer.directory}", flush=True)


if __name__ == "__main__":
    main(Args().parse_args())
