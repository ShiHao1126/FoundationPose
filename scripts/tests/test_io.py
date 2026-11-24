from dataclasses import asdict
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from mesh_pose.contracts import Config, TrackingResult
from mesh_pose.sequence import FolderSequence, RunWriter


def make_sequence(root):
    (root / "rgb").mkdir()
    (root / "depth").mkdir()
    K = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]])
    np.savetxt(root / "cam_K.txt", K)
    for i in range(2):
        cv2.imwrite(str(root / "rgb" / f"{i:04d}.png"), np.full((20, 20, 3), [1, 2, 3], np.uint8))
        cv2.imwrite(str(root / "depth" / f"{i:04d}.png"), np.full((20, 20), 500, np.uint16))


def test_recording_units_rgb_order_and_output_round_trip(tmp_path):
    make_sequence(tmp_path)
    source = FolderSequence(tmp_path, 0.001, 30)
    frame = source.frame(0)
    assert frame.depth_m[0, 0] == 0.5
    assert frame.rgb[0, 0].tolist() == [3, 2, 1]
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    writer = RunWriter(tmp_path / "runs", config, {"time_source": source.time_source}, 30, True)
    try:
        writer.write(frame, TrackingResult(0, 0, "LOST", "missing_depth", None, None, 0.001))
    finally:
        writer.close()
    data = json.loads((writer.directory / "poses.jsonl").read_text())
    assert data["valid"] is False and data["T_camera_object"] is None
    completion = json.loads((writer.directory / "completion.json").read_text())
    assert completion["status"] == "interrupted_or_failed" and completion["frames_written"] == 1
    video = cv2.VideoCapture(str(writer.directory / "preview.mp4"))
    try:
        assert video.read()[0]
    finally:
        video.release()


def test_missing_pair_fails_instead_of_zipping_unrelated_frames(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "depth" / "0001.png").rename(tmp_path / "depth" / "different.png")
    with pytest.raises(FileNotFoundError, match="synchronized"):
        FolderSequence(tmp_path, 0.001, 30)


def test_timestamp_map_and_strict_config(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "timestamps.json").write_text('{"0000": 1.0, "0001": 0.0}')
    with pytest.raises(ValueError, match="increasing"):
        FolderSequence(tmp_path, 0.001, 30)
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    raw = asdict(config)
    raw["tracking"]["confirmation_frames"] = 1.5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="integer"):
        Config.load(path)

