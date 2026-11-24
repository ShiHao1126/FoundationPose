"""Camera contract checks with fake SDK frames; no connected device is accessed."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mesh_pose.realsense import RealSenseConfig, RealSenseSource


class VideoFrame:
    def __init__(self, number, timestamp_ms, data):
        self.number, self.timestamp_ms, self.data = number, timestamp_ms, data

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.timestamp_ms

    def get_frame_timestamp_domain(self):
        return "system_time"

    def get_data(self):
        return self.data


def frameset(number, timestamp_ms):
    color = VideoFrame(number, timestamp_ms, np.ones((12, 16, 3), np.uint8))
    depth = VideoFrame(number + 1, timestamp_ms - 10, np.full((12, 16), 500, np.uint16))
    return SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)


def source_with_frames(frames):
    source = RealSenseSource(RealSenseConfig("fake", 16, 12, 30, 100, 1))
    pending = iter(frames)
    stopped = []
    source._pipeline = SimpleNamespace(wait_for_frames=lambda timeout: next(pending), stop=lambda: stopped.append(True))
    source._align = SimpleNamespace(process=lambda frames: frames)
    source.depth_scale_m = 0.001
    source.K = np.array([[20, 0, 8], [0, 20, 6], [0, 0, 1]], np.float32)
    return source, stopped


def test_reused_color_exposures_are_skipped_and_buffers_are_owned():
    first = frameset(1, 1000)
    source, _ = source_with_frames([first, frameset(1, 1000), frameset(2, 1033)])
    try:
        a = source.read()
        first.get_color_frame().data[:] = 99
        b = source.read()
        assert a.rgb[0, 0, 0] == 1 and a.depth_m[0, 0] == 0.5
        assert b.sequence == 2 and b.timestamp_s > a.timestamp_s
        assert source.duplicate_color_frames == 1
        assert source.last_frame_metadata["timestamp_skew_ms"] == 10
    finally:
        source.close()


def test_dropped_exposures_are_counted_and_close_is_idempotent():
    source, stopped = source_with_frames([frameset(1, 1000), frameset(4, 1100)])
    source.read()
    source.read()
    assert source.dropped_color_frames == 2
    source.close()
    source.close()
    assert len(stopped) == 1
    with pytest.raises(RuntimeError, match="start"):
        source.read()


def test_backwards_sensor_timestamp_is_rejected():
    source, _ = source_with_frames([frameset(1, 1000), frameset(2, 999)])
    try:
        source.read()
        with pytest.raises(RuntimeError, match="backwards"):
            source.read()
    finally:
        source.close()


def test_clock_domain_change_is_rejected():
    changed = frameset(2, 1033)
    changed.get_color_frame().get_frame_timestamp_domain = lambda: "hardware_clock"
    source, _ = source_with_frames([frameset(1, 1000), changed])
    try:
        source.read()
        with pytest.raises(RuntimeError, match="domain changed"):
            source.read()
    finally:
        source.close()


def test_camera_selection_must_be_explicit(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"serial": "", "width": 640, "height": 480, "fps": 30,
                                "timeout_ms": 3000, "warmup_frames": 30}))
    with pytest.raises(ValueError, match="serial"):
        RealSenseConfig.load(path)


@pytest.mark.parametrize("fail_restore", [False, True])
def test_color_settings_restore_all_options_and_always_release_camera(fail_restore):
    source, stopped = source_with_frames([])
    restored = []
    def set_option(option, value):
        restored.append((option, value))
        if option == "exposure" and fail_restore:
            raise RuntimeError("device option write failed")
    source._color_sensor = SimpleNamespace(set_option=set_option)
    source._saved_color_options = [("exposure", 156), ("gain", 64), ("auto", 1)]
    if fail_restore:
        with pytest.raises(RuntimeError, match="restored"):
            source.close()
    else:
        source.close()
    assert restored == [("exposure", 156), ("gain", 64), ("auto", 1)] and len(stopped) == 1
    source.close()
    assert len(stopped) == 1
