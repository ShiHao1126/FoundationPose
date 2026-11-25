"""Live scheduling/freshness/cleanup without a camera, GUI, or GPU."""

from dataclasses import replace
from pathlib import Path
from threading import get_ident
from time import monotonic, sleep
import json

import numpy as np
import pytest

from mesh_pose.contracts import Config, TrackingResult
from mesh_pose.live import Estimate, LiveSession, ReceivedFrame
from mesh_pose.sequence import RunWriter
from mesh_pose.viewer import live_status, preview_lut, preview_rgb
from test_tracking import Backend, frame


class Source:
    def __init__(self, fail_after=None):
        self.index = 0
        self.closed = False
        self.owner = None
        self.fail_after = fail_after
        self.metadata = {"device": "fake"}
        self.last_frame_metadata = {}
        self.dropped_color_frames = 0
        self.duplicate_color_frames = 0

    def __enter__(self):
        self.owner = get_ident()
        return self

    def read(self):
        assert self.owner == get_ident()
        sleep(.005)
        if self.fail_after is not None and self.index >= self.fail_after:
            raise RuntimeError("camera disconnected")
        self.index += 1
        return frame(self.index)

    def __exit__(self, *args):
        assert self.owner == get_ident()
        self.closed = True


class SlowBackend(Backend):
    def __init__(self):
        super().__init__()
        self.owner = get_ident()
        self.object_bounds_m = np.array([[0, 0, 0], [.07, .062, .07]])
        self.runtime = {"fake": ["fake"]}
        self.T_centered_object = np.eye(4)

    def register(self, frame, mask):
        assert self.owner == get_ident()
        sleep(.12)
        return super().register(frame, mask)

    def close(self):
        assert self.owner == get_ident()
        super().close()


def wait_until(predicate, timeout=3):
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() > deadline:
            raise AssertionError("timed out")
        sleep(.01)


@pytest.mark.parametrize("fault", [None, "camera", "backend"])
def test_live_latest_frames_thread_ownership_and_shutdown(tmp_path, fault):
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    source = Source(fail_after=40 if fault == "camera" else None)
    created = []
    def factory(K):
        if fault == "backend":
            raise RuntimeError("GPU unavailable")
        backend = SlowBackend()
        created.append(backend)
        return backend
    writer = RunWriter(tmp_path, config, {}, 5, False)
    session = LiveSession(source, factory, config, writer)
    session.start()
    try:
        if fault:
            wait_until(lambda: session.snapshot()["error"] is not None)
        else:
            wait_until(lambda: session.snapshot()["estimate"] is not None
                       and session.snapshot()["estimate"].result.state == "TRACKING")
            assert source.index > 10  # Camera kept reading throughout slow registration.
            assert session.snapshot()["inference_skipped"] > 10
            assert created[0].owner != source.owner and created[0].owner != get_ident()
    finally:
        session.close(3)
    assert source.closed and session.finished
    assert not created or created[0].closed
    completion = json.loads((writer.directory / "completion.json").read_text())
    assert completion["status"] == ("interrupted_or_failed" if fault else "complete")
    session.close(3)


def test_live_status_invalidates_old_pose_fault_and_stop():
    result = TrackingResult(1, 1 / 30, "TRACKING", "tracked", np.eye(4), None, .2)
    source = ReceivedFrame(frame(1), 100.0, {}, 1)
    estimate = Estimate(source, result, 100.2)
    snapshot = {"estimate": estimate, "frame": source, "error": None,
                "phase": "refining pose", "capture": {}, "inference_skipped": 0}
    assert live_status(snapshot, 100.3, .5, False)["valid_now"]
    for now, error, stopping in [(100.6, None, False), (100.3, "camera failed", False), (100.3, None, True)]:
        status = live_status({**snapshot, "error": error}, now, .5, stopping)
        assert not status["valid_now"] and status["T_camera_object"] is None
    assert not replace(estimate, result=replace(result, T_camera_object=None)).is_current(100.3, .5)


def test_preview_denoise_preserves_input_and_silhouette():
    rgb = np.full((40, 60, 3), 20, np.uint8)
    rgb[:, 30:] = 215
    rgb[10, 10] = 27
    original = rgb.copy()
    lut = preview_lut(1.7)
    smoothed = preview_rgb(rgb, lut, True)
    assert np.array_equal(rgb, original)
    assert np.array_equal(preview_rgb(rgb, lut, False), lut[original])
    assert smoothed[10, 10, 0] < lut[27]
    assert np.array_equal(smoothed[20, 29:31], lut[original[20, 29:31]])

def test_capture_history_is_sampled_causal_and_rejects_gaps():
    from mesh_pose.frame_history import FrameHistory

    history = FrameHistory()
    for i in range(25):
        history.append(frame(i))
    samples = history.between(frame(1), frame(11))
    assert [sample.sequence for sample in samples] == [3, 6, 9, 11]
    history.append(frame(35))
    assert history.between(frame(15), frame(36)) == ()
    history.clear()
    assert history.between(frame(1), frame(11)) == ()
