"""Summarize a live diagnostic run; physical occlusion evidence needs image review."""

from pathlib import Path
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]


class Args(Tap):
    run: Path  # One runs/live/<timestamp> directory.


def analyze(directory: Path) -> dict:
    import json
    import numpy as np

    poses = [json.loads(line) for line in (directory / "poses.jsonl").read_text().splitlines()]
    times = {record["sequence"]: record for record in map(
        json.loads, (directory / "timing.jsonl").read_text().splitlines())}
    if not poses:
        raise ValueError("No processed frames")
    assert all(b["sequence"] > a["sequence"] and b["timestamp_s"] > a["timestamp_s"]
               for a, b in zip(poses, poses[1:])), "Camera chronology violated"
    assert all(p["valid"] == (p["T_camera_object"] is not None) == (p["state"] == "TRACKING")
               for p in poses), "Invalid state published a pose"
    transitions, recoveries, segments = [], [], []
    segment = []
    previous = None
    lost = None
    had_tracking = False
    for pose in poses:
        timing = times[pose["sequence"]]
        if pose["valid"]:
            segment.append(pose)
        elif segment:
            segments.append(segment)
            segment = []
        if pose["state"] != previous:
            transitions.append({"sequence": pose["sequence"], "state": pose["state"],
                                "reason": pose["reason"], "camera_timestamp_s": pose["timestamp_s"],
                                "available_monotonic_s": timing["completed_monotonic_s"]})
            previous = pose["state"]
        if pose["valid"]:
            if lost is not None:
                recoveries.append({
                    "unavailable_sequence": lost["sequence"], "initial_state": lost["state"], "initial_reason": lost["reason"],
                    "recovered_sequence": pose["sequence"],
                    "unavailable_to_recovery_available_s": timing["completed_monotonic_s"] - times[lost["sequence"]]["completed_monotonic_s"],
                    "unavailable_to_recovery_camera_s": pose["timestamp_s"] - lost["timestamp_s"],
                    "recovered_arrival_age_s": timing["arrival_to_result_s"],
                })
                lost = None
            had_tracking = True
        elif had_tracking and lost is None:
            lost = pose
    if segment:
        segments.append(segment)
    segment_rates = []
    for segment in segments:
        if len(segment) < 2:
            continue
        completed = [times[p["sequence"]]["completed_monotonic_s"] for p in segment]
        span = completed[-1] - completed[0]
        segment_rates.append({
            "first_sequence": segment[0]["sequence"], "last_sequence": segment[-1]["sequence"],
            "valid_frames": len(segment), "span_s": span,
            "valid_publication_hz": (len(segment) - 1) / span,
            "completion_interval_p95_s": float(np.percentile(np.diff(completed), 95)),
        })
    tracking = [p["elapsed_s"] for p in poses if p["valid"]]
    status = json.loads((directory / "status.json").read_text())
    report = {
        "run": str(directory.resolve()), "source": "live RealSense and live GPU inference",
        "processed_frames": len(poses), "valid_frames": len(tracking),
        "camera_span_s": poses[-1]["timestamp_s"] - poses[0]["timestamp_s"],
        "tracking_median_s": float(np.median(tracking)) if tracking else None,
        "tracking_p95_s": float(np.percentile(tracking, 95)) if tracking else None,
        "transitions": transitions, "recoveries": recoveries,
        "continuous_tracking_segments": sorted(segment_rates, key=lambda s: s["span_s"], reverse=True),
        "capture": status["capture"], "inference_skipped": status["inference_skipped"],
        "error": status["error"],
        "measurement_scope": "Recovery interval includes time hidden and search; not measured from uncovering. "
                             "State changes alone do not prove physical occlusion. No independent pose ground truth.",
    }
    (directory / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(analyze(Args().parse_args().run), indent=2))
