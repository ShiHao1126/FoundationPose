"""Latest-frame camera/inference workers; the Tracker alone owns pose state.

Camera capture never waits for inference, disk, or the GUI. OpenGL resources are
constructed, used and closed on the inference thread. Snapshots own their arrays
and are treated as read-only after publication.
"""

from dataclasses import dataclass
from collections import deque
from threading import Condition, Event, Thread
from time import monotonic
import json
import logging

import cv2

from .contracts import RGBDFrame, TrackingResult
from .sequence import RunWriter
from .tracker import Tracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReceivedFrame:
    frame: RGBDFrame
    received_s: float  # Host monotonic clock, after SDK read/alignment.
    pairing: dict
    count: int


@dataclass(frozen=True)
class Estimate:
    source: ReceivedFrame
    result: TrackingResult
    completed_s: float

    def is_current(self, now: float, max_age_s: float) -> bool:
        return self.result.T_camera_object is not None and 0 <= now - self.source.received_s <= max_age_s


class LiveSession:
    def __init__(self, source, backend_factory, config, writer: RunWriter):
        self.source = source
        self.backend_factory = backend_factory
        self.config = config
        self.writer = writer
        self.stop = Event()
        self._condition = Condition()
        self._latest = None
        self._estimate = None
        self._error = None
        self._phase = "opening camera"
        self._camera_metadata = None
        self._bounds = None
        self._capture_stats = {}
        self._inference_skipped = 0
        self._inference_stats = {}
        self._threads = []

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("Live session already started")
        self._threads = [Thread(target=self._capture, name="realsense"),
                         Thread(target=self._infer, name="pose-inference")]
        for thread in self._threads:
            thread.start()

    def request_stop(self) -> None:
        self.stop.set()
        with self._condition:
            self._condition.notify_all()

    @property
    def finished(self) -> bool:
        return bool(self._threads) and not any(t.is_alive() for t in self._threads)

    def snapshot(self) -> dict:
        with self._condition:
            return {"frame": self._latest, "estimate": self._estimate, "error": self._error,
                    "phase": self._phase, "camera": self._camera_metadata, "bounds": self._bounds,
                    "capture": self._capture_stats.copy(), "inference_skipped": self._inference_skipped,
                    "inference": self._inference_stats.copy()}

    def fail(self, component: str) -> None:
        """Latch the active exception, then request every owner to stop."""
        logger.exception("%s failed", component)
        import traceback
        with self._condition:
            if self._error is None:
                self._error = component + ": " + traceback.format_exc().strip().splitlines()[-1]
        self.request_stop()

    def _capture(self) -> None:
        count = 0
        first = None
        try:
            with self.source:
                with self._condition:
                    self._camera_metadata = self.source.metadata.copy()
                while not self.stop.is_set():
                    frame = self.source.read()
                    received = monotonic()
                    count += 1
                    if first is None:
                        first = (frame.sequence, frame.timestamp_s, received)
                    item = ReceivedFrame(frame, received, self.source.last_frame_metadata.copy(), count)
                    with self._condition:
                        self._latest = item  # One slot: no inference backlog.
                        sensor_span = frame.timestamp_s - first[1]
                        host_span = received - first[2]
                        self._capture_stats = {
                            "frames_received": count,
                            "sensor_fps": (frame.sequence - first[0]) / sensor_span if sensor_span else 0,
                            "received_fps": (count - 1) / host_span if host_span > 0 else 0,
                            "sdk_dropped_color": self.source.dropped_color_frames,
                            "sdk_duplicate_color": self.source.duplicate_color_frames,
                        }
                        self._condition.notify_all()
        except Exception:
            self.fail("camera")

    def _next(self, after_sequence: int) -> ReceivedFrame | None:
        with self._condition:
            self._condition.wait_for(lambda: self.stop.is_set() or (
                self._latest is not None and self._latest.frame.sequence > after_sequence))
            return None if self.stop.is_set() else self._latest

    def _infer(self) -> None:
        tracker = None
        previous = None
        previous_state = None
        recent = deque()
        try:
            item = self._next(-1)
            if item is None:
                return
            with self._condition:
                self._phase = "loading GPU models and CAD templates"
            backend = self.backend_factory(item.frame.K)
            tracker = Tracker(backend, self.config)
            with self._condition:
                self._bounds = backend.object_bounds_m.copy()
            (self.writer.directory / "camera.json").write_text(json.dumps(
                self.snapshot()["camera"], indent=2), encoding="utf-8")
            (self.writer.directory / "backend.json").write_text(json.dumps({
                "runtime_providers": backend.runtime,
                "T_centered_object": backend.T_centered_object.tolist(),
            }, indent=2), encoding="utf-8")
            with (self.writer.directory / "timing.jsonl").open("x", encoding="utf-8") as timing:
                rate_started = monotonic()
                while not self.stop.is_set():
                    item = self._next(previous.frame.sequence if previous else -1)
                    if item is None:
                        break
                    with self._condition:
                        self._phase = "recovery / global search" if tracker.state in {"SEARCHING", "LOST"} else "refining pose"
                        if previous is not None:
                            self._inference_skipped += item.count - previous.count - 1
                    started = monotonic()
                    result = tracker.step(item.frame, cooperative=True)
                    estimate = Estimate(item, result, monotonic())
                    recent.append((estimate.completed_s, result.T_camera_object is not None))
                    while recent and recent[0][0] < estimate.completed_s - 5:
                        recent.popleft()
                    span = min(5.0, estimate.completed_s - rate_started)
                    with self._condition:
                        self._estimate = estimate
                        self._inference_stats = {
                            "window_s": span, "processed_hz": len(recent) / span,
                            "valid_pose_hz": sum(valid for _, valid in recent) / span,
                        }
                    self.writer.write(item.frame, result)
                    timing.write(json.dumps({
                        "sequence": result.sequence, "received_monotonic_s": item.received_s,
                        "started_monotonic_s": started, "completed_monotonic_s": estimate.completed_s,
                        "arrival_to_result_s": estimate.completed_s - item.received_s,
                        "state": result.state, "reason": result.reason,
                        "inference_skipped": self._inference_skipped, "pairing": item.pairing,
                    }) + "\n")
                    timing.flush()
                    if result.state != previous_state:
                        logger.info("frame=%d %s: %s (%.3fs)", result.sequence, result.state, result.reason, result.elapsed_s)
                        self._save_transition(item, result)
                        previous_state = result.state
                    elif result.reason == "no_verified_candidate":
                        self._save_transition(item, result)
                    previous = item
        except Exception:
            self.fail("inference")
        finally:
            if tracker is not None:
                try:
                    tracker.close()
                except Exception:
                    self.fail("backend close")

    def _save_transition(self, item: ReceivedFrame, result: TrackingResult) -> None:
        # Small diagnostic evidence only, not a full training-data recorder.
        directory = self.writer.directory / "transitions"
        directory.mkdir(exist_ok=True)
        stem = f"{result.sequence:08d}_{result.state}"
        if not cv2.imwrite(str(directory / f"{stem}.png"), cv2.cvtColor(item.frame.rgb, cv2.COLOR_RGB2BGR)):
            raise OSError("Could not save transition image")
        import numpy as np
        np.savez_compressed(directory / f"{stem}.npz", depth_m=item.frame.depth_m, K=item.frame.K)

    def close(self, timeout_s: float = 60) -> None:
        self.request_stop()
        deadline = monotonic() + timeout_s
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))
        if not self.finished and self._threads:
            raise TimeoutError("Workers still stopping; do not start a second camera session")
        if self._error is None:
            self.writer.mark_complete()
        self.writer.close()
