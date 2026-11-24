"""Recorded RGB-D input and append-only, bounded-memory run output."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np

from .contracts import Config, RGBDFrame, TrackingResult


class FolderSequence:
    """Matching rgb/<stem>.png + depth/<stem>.png; never reads masks or annotations."""

    def __init__(self, root: Path, depth_scale_m: float, fps: float):
        if not np.isfinite(depth_scale_m) or depth_scale_m <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("depth_scale_m and fps must be positive")
        self.root = root
        self.depth_scale_m = depth_scale_m
        self.fps = fps
        self.K = np.loadtxt(root / "cam_K.txt", dtype=np.float32).reshape(3, 3)
        self.rgb_files = sorted((root / "rgb").glob("*.png"))
        if not self.rgb_files:
            raise FileNotFoundError(f"No PNG RGB frames in {root / 'rgb'}")
        for path in self.rgb_files:
            if not (root / "depth" / path.name).is_file():
                raise FileNotFoundError(f"Missing synchronized depth for {path.name}")
        time_file = root / "timestamps.json"
        self.timestamps = json.loads(time_file.read_text(encoding="utf-8")) if time_file.exists() else None
        self.time_source = "timestamps.json" if self.timestamps is not None else "synthetic_from_fps"
        if self.timestamps is not None:
            times = [float(self.timestamps[p.stem]) for p in self.rgb_files]
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("timestamps.json must contain finite, increasing seconds for every RGB stem")

    def __len__(self) -> int:
        return len(self.rgb_files)

    def frame(self, index: int) -> RGBDFrame:
        path = self.rgb_files[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(self.root / "depth" / path.name), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw_depth is None:
            raise ValueError(f"Cannot decode RGB-D frame {path.name}")
        frame = RGBDFrame(index, float(self.timestamps[path.stem]) if self.timestamps is not None else index / self.fps,
                          cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                          raw_depth.astype(np.float32) * self.depth_scale_m, self.K)
        frame.validate()
        return frame


class RunWriter:
    def __init__(self, root: Path, config: Config, metadata: dict, fps: float, save_video: bool):
        self.directory = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "run.json").write_text(json.dumps(
            {"schema_version": 1, "config": asdict(config), **metadata}, indent=2), encoding="utf-8")
        self._file = (self.directory / "poses.jsonl").open("x", encoding="utf-8")
        self._video = None
        self._fps = fps
        self._save_video = save_video
        self._frames_written = 0
        self._completed = False
        self._closed = False

    def write(self, frame: RGBDFrame, result: TrackingResult) -> None:
        self._file.write(json.dumps(result.to_record(), allow_nan=False) + "\n")
        self._file.flush()
        self._frames_written += 1
        if self._save_video:
            canvas = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if result.T_camera_object is not None:
                pose = result.T_camera_object
                rvec, _ = cv2.Rodrigues(pose[:3, :3])
                cv2.drawFrameAxes(canvas, frame.K, np.zeros(5), rvec, pose[:3, 3], 0.03, 2)
            color = (80, 220, 80) if result.T_camera_object is not None else (40, 80, 240)
            cv2.putText(canvas, f"{result.sequence} {result.state}: {result.reason}", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            if self._video is None:
                height, width = canvas.shape[:2]
                self._video = cv2.VideoWriter(str(self.directory / "preview.mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (width, height))
                if not self._video.isOpened():
                    raise RuntimeError("Cannot open preview video writer")
            self._video.write(canvas)

    def mark_complete(self) -> None:
        self._completed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()
        if self._video is not None:
            self._video.release()
        (self.directory / "completion.json").write_text(json.dumps({
            "status": "complete" if self._completed else "interrupted_or_failed",
            "frames_written": self._frames_written,
        }, indent=2), encoding="utf-8")

