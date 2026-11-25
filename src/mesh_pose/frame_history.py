"""Short, bounded RGB-D history shared by the capture and pose workers."""

from collections import deque
from threading import Lock

from .contracts import RGBDFrame


class FrameHistory:
    """Keep roughly 10 Hz of owned camera snapshots; never queue inference work."""

    def __init__(self):
        self._frames = deque()
        self._lock = Lock()
        self._bytes = 0

    def append(self, frame: RGBDFrame) -> None:
        with self._lock:
            if self._frames and frame.timestamp_s - self._frames[-1].timestamp_s < 0.09:
                return
            self._frames.append(frame)
            self._bytes += frame.rgb.nbytes + frame.depth_m.nbytes
            while self._frames and (self._bytes > 256 * 1024**2 or
                                    frame.timestamp_s - self._frames[0].timestamp_s > 12):
                old = self._frames.popleft()
                self._bytes -= old.rgb.nbytes + old.depth_m.nbytes

    def between(self, source: RGBDFrame, current: RGBDFrame) -> tuple[RGBDFrame, ...]:
        """Return intermediate observations followed by the exact verification frame."""
        with self._lock:
            frames = tuple(f for f in self._frames if source.sequence < f.sequence < current.sequence)
        if not frames:
            return ()
        previous = source
        for frame in (*frames, current):
            if not 0 < frame.timestamp_s - previous.timestamp_s <= 0.25:
                return ()
            previous = frame
        return (*frames, current)

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._bytes = 0
