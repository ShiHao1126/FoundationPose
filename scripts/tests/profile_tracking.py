"""Isolated saved-frame performance diagnosis; opens no camera and changes no tuning."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]


class Args(Tap):
    mode: str = "tracking"  # tracking, registration, parity, or acquisition (saved failure replay).
    run: Path = ROOT / "runs" / "live" / "20261124_135208_204711"
    sequence: int = 3235  # Automatically acquired TRACKING transition, with saved RGB-D.
    followup_sequences: list[int] = [760, 765]  # Distinct saved sensor frames for acquisition replay.
    samples: int = 20
    warmup: int = 3
    output: Path = ROOT / "runs" / "profiling"


def main(args: Args):
    import cProfile
    from dataclasses import replace
    from datetime import datetime
    from functools import wraps
    import json
    import logging
    import pstats
    from time import perf_counter

    import cv2
    import numpy as np
    import mesh_pose.backend as backend_module
    from mesh_pose.backend import SpatialHubBackend
    from mesh_pose.contracts import Config, RGBDFrame
    from mesh_pose.geometry import verify_depth

    if args.samples < 1 or args.warmup < 0 or args.mode not in {"tracking", "registration", "parity", "acquisition"}:
        raise ValueError("samples must be positive and warmup nonnegative")
    out = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, filename=out / "runtime.log",
                        format="%(asctime)s %(levelname)s %(message)s")
    record = next(json.loads(line) for line in (args.run / "poses.jsonl").read_text().splitlines()
                  if json.loads(line)["sequence"] == args.sequence)
    path = args.run / "transitions" / f"{args.sequence:08d}_{record['state']}"
    data = np.load(str(path) + ".npz")
    frame = RGBDFrame(args.sequence, record["timestamp_s"],
                      cv2.cvtColor(cv2.imread(str(path) + ".png"), cv2.COLOR_BGR2RGB),
                      data["depth_m"], data["K"])
    frame.validate()
    config = Config.load(ROOT / "configs" / "tracker.json")
    backend = SpatialHubBackend(ROOT / "data/meshes/cover/cover.stl", "mm", frame.K,
                                config.backend, config.geometry, ROOT / "configs/models.json", ROOT / "cache")
    try:
        if args.mode == "acquisition":
            report = replay_acquisition(backend, frame, config, args, out)
            (out / "acquisition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            print(f"Saved acquisition replay: {out}", flush=True)
            if not report["passed"]:
                raise RuntimeError("Saved acquisition failure still does not reach TRACKING")
            return
        if args.mode == "parity":
            report = compare_paths(backend, frame, record, config)
            (out / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            print(f"Saved equivalence checks: {out}", flush=True)
            if not report["passed"]:
                raise RuntimeError("Optimized path differs beyond numerical tolerance")
            return
        seed = (np.array(record["T_camera_object"], np.float32) @ np.linalg.inv(backend.T_centered_object)
                if args.mode == "tracking" else None)
        per_frame = []
        stage = {}
        def wrap(obj, name, label):
            original = getattr(obj, name)
            @wraps(original)
            def timed(*a, **kw):
                key = label(*a, **kw) if callable(label) else label
                start = perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    duration, count = stage.get(key, (0., 0))
                    stage[key] = (duration + perf_counter() - start, count + 1)
            setattr(obj, name, timed)
        wrap(backend.estimator, "track", "spatialhub_track_inclusive")
        wrap(backend.estimator, "register", "spatialhub_register_inclusive")
        wrap(backend, "register", "candidate_register_inclusive")
        wrap(backend, "detect", "automatic_detection")
        wrap(backend.estimator.depth_filter, "apply", "depth_filter")
        wrap(backend.estimator.refiner, "predict", "refiner_inclusive" if args.mode == "tracking" else
             lambda *a, **kw: "coarse_refiner_inclusive" if len(a[3]) > 1 else "fine_refiner_inclusive")
        wrap(backend.estimator.refiner.ort_session, "run", "onnx_session_run")
        wrap(backend.estimator.scorer, "predict", "scorer_inclusive")
        wrap(backend.estimator.scorer.ort_session, "run", "scorer_onnx_run")
        wrap(backend.estimator.renderer, "render", "all_renderer_calls_inclusive")
        wrap(backend.estimator.renderer.engine, "read_tiles", "render_readback")
        wrap(backend, "render_depth", "icp_verification_render_inclusive")
        wrap(backend_module, "refine_depth_pose", "added_icp_inclusive")

        def one_step():
            if args.mode == "registration":
                from mesh_pose.tracker import Tracker
                tracker = Tracker(backend, config)
                for index in range(1000):
                    result = tracker.step(replace(frame, sequence=frame.sequence + index,
                                                   timestamp_s=frame.timestamp_s + index / 30), cooperative=True)
                    if result.reason != "global_search_in_progress":
                        return result.quality
                raise RuntimeError("Cooperative registration did not finish")
            pose = backend.track(frame, seed.copy())
            start = perf_counter()
            quality = verify_depth(frame.depth_m, backend.render_depth(frame, pose), config.geometry)
            stage["final_geometry_check_inclusive"] = (perf_counter() - start, 1)
            return quality
        for _ in range(args.warmup):
            one_step()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.samples):
            stage.clear()
            start = perf_counter()
            quality = one_step()
            per_frame.append({"total_s": perf_counter() - start, "stages": stage.copy(),
                              "quality_accepted": bool(quality and quality.accepted)})
        profiler.disable()
        with (out / "cprofile.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("tottime").print_stats(25)
        timings = {label: {"median_ms_per_frame": float(np.median([r["stages"].get(label, (0, 0))[0] for r in per_frame]) * 1000),
                            "mean_calls_per_frame": float(np.mean([r["stages"].get(label, (0, 0))[1] for r in per_frame]))}
                   for label in per_frame[0]["stages"]}

        comparisons = {}
        for iterations in ((1, 2) if args.mode == "tracking" else ()):
            durations = []
            for _ in range(args.samples):
                backend.estimator.current_pose = seed.copy()
                start = perf_counter()
                backend.estimator.track(frame.rgb, frame.depth_m, depth_unit="m", iteration=iterations, verify_conf=False)
                durations.append(perf_counter() - start)
            comparisons[f"spatialhub_without_added_icp_{iterations}_iter_ms"] = float(np.median(durations) * 1000)
        durations = []
        for _ in range(5 if args.mode == "tracking" else 0):
            start = perf_counter()
            backend.detect(frame)
            durations.append(perf_counter() - start)
        if durations:
            comparisons["automatic_full_image_detection_ms"] = float(np.median(durations) * 1000)
        report = {"mode": args.mode, "rotation_hypotheses_per_candidate": len(backend.estimator.rot_grid),
                  "input_run": str(args.run), "input_sequence": args.sequence, "samples": args.samples,
                  "test_type": "Saved fixed RGB-D; tracking reuses automatic seed, registration searches from scratch; no camera/UI/disk in timed loop",
                  "gl": {k: backend.estimator.glctx.info[k] for k in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")},
                  "providers": backend.runtime,
                  "full_step_median_ms": float(np.median([r["total_s"] for r in per_frame]) * 1000),
                  "stages_inclusive_do_not_sum": timings, "isolated_comparisons": comparisons,
                  "accepted_samples": sum(r["quality_accepted"] for r in per_frame)}
        (out / "profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Saved profile: {out}", flush=True)
    finally:
        backend.close()


def compare_paths(backend, frame, record, config):
    """Numerical checks on real GPU, without opening a camera or changing thresholds."""
    import cv2
    import numpy as np
    from time import perf_counter
    from mesh_pose.geometry import pose_difference, verify_depth

    seed = np.array(record["T_camera_object"], np.float32) @ np.linalg.inv(backend.T_centered_object)
    optimized = backend.render_depth
    def reference(image, pose):
        h, w = image.depth_m.shape
        return backend.estimator.renderer.render(
            intrinsics=image.K, H=h, W=w, ob_in_cams=pose[None], output_size=(w, h))[1][0]
    rng = np.random.default_rng(42)
    depth_errors, silhouette_errors, pose_errors = [], [], []
    for index in range(40):
        pose = seed.copy()
        pose[:3, :3] = cv2.Rodrigues(rng.normal(size=3) * .3)[0] @ seed[:3, :3]
        pose[:3, 3] += rng.normal(size=3) * .025
        if index >= 30:
            pose[0, 3] = (-1 if index % 2 else 1) * pose[2, 3] * .52
        old, new = reference(frame, pose), optimized(frame, pose)
        silhouette_errors.append(int(np.count_nonzero((old > 0) != (new > 0))))
        depth_errors.append(float(np.max(np.abs(old - new))))
    for _ in range(8):
        pose = seed.copy()
        pose[:3, :3] = cv2.Rodrigues(rng.normal(size=3) * .02)[0] @ seed[:3, :3]
        pose[:3, 3] += rng.normal(size=3) * .002
        try:
            backend.render_depth = reference
            old = backend.track(frame, pose.copy())
        finally:
            backend.render_depth = optimized
        new = backend.track(frame, pose.copy())
        distance, angle = pose_difference(old, new)
        pose_errors.append({"translation_m": distance, "rotation_deg": angle,
                            "old_accepted": verify_depth(frame.depth_m, reference(frame, old), config.geometry).accepted,
                            "new_accepted": verify_depth(frame.depth_m, optimized(frame, new), config.geometry).accepted})
    registration = []
    for candidate in backend.detect(frame):
        old = backend.register(frame, candidate.mask)
        steps = backend.register_steps(frame, candidate.mask)
        durations = []
        while True:
            start = perf_counter()
            try:
                next(steps)
            except StopIteration as done:
                new = done.value
                durations.append(perf_counter() - start)
                break
            durations.append(perf_counter() - start)
        distance, angle = pose_difference(old, new)
        registration.append({"translation_m": distance, "rotation_deg": angle,
                             "steps": len(durations), "total_s": sum(durations),
                             "max_uninterrupted_step_s": max(durations)})
    return {"passed": max(silhouette_errors) == 0 and max(depth_errors) < 1e-6
            and all(p["translation_m"] < 1e-4 and p["rotation_deg"] < .1
                    and p["old_accepted"] == p["new_accepted"] for p in pose_errors)
            and all(p["translation_m"] < 1e-4 and p["rotation_deg"] < .1 for p in registration),
            "renderer_poses": 40, "max_depth_error_m": max(depth_errors),
            "max_silhouette_pixel_difference": max(silhouette_errors),
            "tracking_pose_comparisons": pose_errors, "registration_comparisons": registration,
            "accuracy_ground_truth": False}


def replay_acquisition(backend, first_frame, config, args, out):
    import cv2
    import json
    import numpy as np
    from mesh_pose.contracts import RGBDFrame
    from mesh_pose.tracker import Tracker
    from mesh_pose.viewer import annotate_pose

    records = {r["sequence"]: r for r in map(json.loads, (args.run / "poses.jsonl").read_text().splitlines())}
    tracker = Tracker(backend, config)
    results = [tracker.step(first_frame)]
    for sequence in args.followup_sequences:
        record = records[sequence]
        path = args.run / "transitions" / f"{sequence:08d}_{record['state']}"
        data = np.load(str(path) + ".npz")
        frame = RGBDFrame(sequence, record["timestamp_s"],
                          cv2.cvtColor(cv2.imread(str(path) + ".png"), cv2.COLOR_BGR2RGB),
                          data["depth_m"], data["K"])
        result = tracker.step(frame)
        results.append(result)
        annotated = annotate_pose(frame.rgb, frame.K, result.T_camera_object, backend.object_bounds_m)
        cv2.imwrite(str(out / f"{sequence}_{result.state}.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return {"passed": results[-1].state == "TRACKING" and all(r.T_camera_object is None for r in results[:-1]),
            "source_run": str(args.run), "test_type": "Real GPU replay of distinct saved RGB-D frames; camera not opened",
            "results": [r.to_record() for r in results], "accuracy_ground_truth": False}


if __name__ == "__main__":
    main(Args().parse_args())
