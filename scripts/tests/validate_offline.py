"""Real GPU inference on a public recording, with an explicit synthetic loss interval."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]


class Args(Tap):
    sequence: Path = ROOT / "data" / "samples" / "mustard0"
    mesh: Path = ROOT / "data" / "samples" / "mustard0" / "mesh" / "textured_simple.obj"
    mesh_unit: str = "m"
    geometry_only_mm: bool = True  # Strip texture and export CAD in mm for the target use case.
    frames_per_segment: int = 8
    output: Path = ROOT / "runs" / "validation"


def main(args: Args) -> None:
    from dataclasses import replace
    import json
    import logging
    import numpy as np
    from mesh_pose.backend import SpatialHubBackend
    from mesh_pose.contracts import Config
    from mesh_pose.sequence import FolderSequence, RunWriter
    from mesh_pose.tracker import Tracker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.frames_per_segment < 2:
        raise ValueError("Need >= 2 frames per segment for confirmation")
    config = Config.load(ROOT / "configs" / "tracker.json")
    source = FolderSequence(args.sequence, 0.001, 30)
    mesh_path, mesh_unit = args.mesh, args.mesh_unit
    if args.geometry_only_mm:
        from spatialhub.utils import load_mesh, scale_mesh
        mesh = scale_mesh(load_mesh(mesh_path), model_unit=mesh_unit)
        mesh.apply_scale(1000.0)
        generated = ROOT / "data" / "samples" / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        mesh_path, mesh_unit = generated / "mustard_geometry_mm.stl", "mm"
        mesh.export(mesh_path)
    backend = SpatialHubBackend(mesh_path, mesh_unit, source.K, config.backend, config.geometry,
                                ROOT / "configs" / "models.json", ROOT / "cache")
    tracker = Tracker(backend, config)
    writer = None
    results = []
    try:
        writer = RunWriter(args.output, config, {"test": "recording_with_synthetic_loss",
            "sequence": str(args.sequence), "runtime_providers": backend.runtime,
            "time_source": "synthetic_from_fps", "mesh": str(mesh_path), "mesh_unit": mesh_unit,
            "geometry_only_mm": args.geometry_only_mm}, 10, True)
        n = args.frames_per_segment
        for index in range(2 * n + 3):
            source_index = index if index < n else max(0, index - 3)
            frame = replace(source.frame(source_index), sequence=index, timestamp_s=index / 30)
            if n <= index < n + 3:
                # Flat, distant background tests free-space rejection; depth is not missing.
                frame = replace(frame, rgb=np.zeros_like(frame.rgb), depth_m=np.full_like(frame.depth_m, 2.0))
            # Complete the cold search first; exercise live scheduling during loss and recovery.
            result = tracker.step(frame, cooperative=index >= n)
            writer.write(frame, result)
            results.append(result)
            print(index, result.state, result.reason, round(result.elapsed_s, 3),
                  result.quality, flush=True)
        initial = any(r.T_camera_object is not None for r in results[:n])
        lost = all(r.T_camera_object is None for r in results[n:n + 3])
        recovered = any(r.T_camera_object is not None for r in results[n + 3:])
        first_valid = next((i for i in range(n + 3, len(results))
                            if results[i].T_camera_object is not None), None)
        summary = {"first_valid_frame_after_return": first_valid - (n + 3) + 1 if recovered else None,
                   "recovery_compute_s": sum(r.elapsed_s for r in results[n + 3:first_valid + 1]) if recovered else None,
                   "automatic_initialization": initial, "no_valid_pose_during_injected_loss": lost,
                   "automatic_recovery": recovered, "hardware_tested": False,
                   "accuracy_measured": False, "frames": len(results)}
        (writer.directory / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Validation:", writer.directory, summary, flush=True)
        if not (initial and lost and recovered):
            raise RuntimeError("Offline recovery acceptance failed; inspect poses.jsonl and preview.mp4")
        writer.mark_complete()
    finally:
        try:
            tracker.close()
        finally:
            if writer is not None:
                writer.close()


if __name__ == "__main__":
    main(Args().parse_args())
