"""Tk operator display; no camera/GPU ownership and no pose state machine."""

from time import monotonic
import itertools
import json
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


def annotate_pose(rgb, K, pose, bounds):
    canvas = rgb.copy()
    if pose is None:
        return canvas
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    # Canvas is RGB, so x is blue here unless drawing in BGR first.
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.drawFrameAxes(bgr, K, np.zeros(5), rvec, pose[:3, 3], 0.03, 2)
    if bounds is not None:
        corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))
        xyz = corners @ pose[:3, :3].T + pose[:3, 3]
        if np.all(xyz[:, 2] > 0):
            uv = xyz @ K.T
            uv = np.round(uv[:, :2] / uv[:, 2:]).astype(int)
            for i in range(8):
                for bit in (1, 2, 4):
                    j = i ^ bit
                    if i < j:
                        cv2.line(bgr, tuple(uv[i]), tuple(uv[j]), (50, 230, 60), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def live_status(snapshot, now: float, max_pose_age_s: float, stopping: bool) -> dict:
    estimate, frame = snapshot["estimate"], snapshot["frame"]
    valid = bool(estimate and estimate.is_current(now, max_pose_age_s)
                 and not snapshot["error"] and not stopping)
    result = estimate.result if estimate else None
    return {
        "schema_version": 1, "updated_monotonic_s": now,
        "phase": "stopping" if stopping else snapshot["phase"], "error": snapshot["error"],
        "state": result.state if result else "STARTING", "reason": result.reason if result else "",
        "valid_now": valid, "max_pose_age_s": max_pose_age_s,
        "pose_age_s": now - estimate.source.received_s if estimate else None,
        "camera_age_s": now - frame.received_s if frame else None,
        "camera_sequence": frame.frame.sequence if frame else None,
        "pose_sequence": result.sequence if result else None,
        "camera_timestamp_s": result.timestamp_s if result else None,
        "T_camera_object": result.T_camera_object.tolist() if valid else None,
        "processing_s": result.elapsed_s if result else None,
        "quality": result.to_record()["quality"] if result else None,
        "capture": snapshot["capture"], "inference_skipped": snapshot["inference_skipped"],
        "inference": snapshot.get("inference", {}),
        "translation_unit": "m", "camera_axes": "x-right,y-down,z-forward",
        "object_frame": "original_mesh_origin_and_axes",
    }


class LiveViewer:
    def __init__(self, session, max_pose_age_s: float, duration_s: float = 0, save_video: bool = False):
        self.session = session
        self.max_pose_age_s = max_pose_age_s
        self.duration_s = duration_s
        self.save_video = save_video
        self.root = tk.Tk()
        self.root.report_callback_exception = self._callback_error
        self.root.title("FoundationPose | Live RealSense + mesh pose")
        self.root.protocol("WM_DELETE_WINDOW", self.session.request_stop)
        self.root.bind("<Escape>", lambda event: self.session.request_stop())
        self.directory = session.writer.directory
        self.started = monotonic()
        self._last_saved = 0
        self._last_video = 0
        self._video = None
        self._video_frames = 0
        self._display_log = None
        self._photos = []
        tk.Label(self.root, text="实时相机（左）  |  位姿对应帧（右，有推理延迟）", font=("Microsoft YaHei UI", 13)).pack()
        self.picture = tk.Label(self.root)
        self.picture.pack()
        self.status = tk.Label(self.root, justify="left", anchor="w", font=("Consolas", 12))
        self.status.pack(fill="x", padx=10)
        tk.Label(self.root, text="位置：原始 mesh 原点，相机坐标 x 右 / y 下 / z 前；轴：红 X、绿 Y、蓝 Z。过期位姿自动隐藏。",
                 font=("Microsoft YaHei UI", 10)).pack()
        tk.Button(self.root, text="停止并释放相机（Esc）", command=session.request_stop).pack(pady=5)

    @staticmethod
    def _callback_error(exc_type, exc_value, traceback):
        # Tk otherwise prints callback failures and leaves a frozen UI running.
        raise exc_value.with_traceback(traceback)

    def _tick(self):
        now = monotonic()
        if (self.duration_s > 0 and now - self.started >= self.duration_s) or (self.directory / "STOP").exists():
            self.session.request_stop()
        snapshot = self.session.snapshot()
        status = live_status(snapshot, now, self.max_pose_age_s, self.session.stop.is_set())
        frame, estimate = snapshot["frame"], snapshot["estimate"]
        if frame is not None:
            left = frame.frame.rgb
            if estimate is not None:
                right = annotate_pose(estimate.source.frame.rgb, estimate.source.frame.K,
                                      estimate.result.T_camera_object if status["valid_now"] else None,
                                      snapshot["bounds"])
            else:
                right = np.zeros_like(left)
            canvas = np.concatenate([left, right], axis=1)
            color = (60, 230, 80) if status["valid_now"] else (250, 130, 40)
            cv2.putText(canvas, "LIVE CAMERA", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 230, 250), 2)
            age = status["pose_age_s"]
            display_state = "STALE" if status["state"] == "TRACKING" and not status["valid_now"] else status["state"]
            label = f'{display_state} / {"VALID" if status["valid_now"] else "NO CURRENT POSE"}'
            cv2.putText(canvas, label, (left.shape[1] + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            if age is not None:
                cv2.putText(canvas, f"pose frame age: {age:.2f}s", (left.shape[1] + 12, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            image = Image.fromarray(canvas)
            max_width = min(1280, self.root.winfo_screenwidth() - 60)
            if image.width > max_width:
                image = image.resize((max_width, round(image.height * max_width / image.width)))
            photo = ImageTk.PhotoImage(image)
            self.picture.configure(image=photo)
            self._photos = [photo]
            if self.save_video and now - self._last_video >= .1:
                if self._video is None:
                    self._video = cv2.VideoWriter(str(self.directory / "live_camera.mp4"),
                                                  cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                                  (canvas.shape[1], canvas.shape[0]))
                    if not self._video.isOpened():
                        raise OSError("Cannot open live camera diagnostic video")
                self._video.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                self._display_log.write(json.dumps({
                    "video_frame": self._video_frames, "monotonic_s": now,
                    "camera_sequence": status["camera_sequence"], "pose_sequence": status["pose_sequence"],
                    "state": status["state"], "valid_now": status["valid_now"],
                    "pose_age_s": status["pose_age_s"],
                }) + "\n")
                self._display_log.flush()
                self._video_frames += 1
                self._last_video = now
        pose = np.array(status["T_camera_object"]) if status["valid_now"] else None
        values = "t [mm]: --       R: --"
        if pose is not None:
            t = pose[:3, 3] * 1000
            values = f"t [mm]: X={t[0]:8.2f}  Y={t[1]:8.2f}  Z={t[2]:8.2f}\nR = " + np.array2string(pose[:3, :3], precision=3, suppress_small=True)
        capture = status["capture"]
        duration = status["processing_s"]
        info = (f'{status["state"]} | {status["phase"]} | {status["reason"]}\n'
                f'Camera {capture.get("received_fps", 0):.1f} Hz | valid poses '
                f'{status["inference"].get("valid_pose_hz", 0):.1f} Hz (5s) | step '
                f'{1000 * duration if duration else 0:.0f} ms | skipped {status["inference_skipped"]} | '
                f'SDK drops {capture.get("sdk_dropped_color", 0)}\n{values}')
        self.status.configure(text=status["error"] or info, fg="red" if status["error"] else "black")
        if now - self._last_saved >= .5 or self.session.finished:
            temporary = self.directory / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.directory / "status.json")
            if frame is not None:
                Image.fromarray(canvas).save(self.directory / "live_preview.jpg", quality=85)
            self._last_saved = now
        if self.session.finished:
            self.root.destroy()
        else:
            self.root.after(33, self._tick)

    def run(self):
        try:
            if self.save_video:
                self._display_log = (self.directory / "display_timing.jsonl").open("x", encoding="utf-8")
            self.root.after(0, self._tick)
            self.root.mainloop()
        finally:
            self.session.request_stop()
            if self._video is not None:
                self._video.release()
            if self._display_log is not None:
                self._display_log.close()
