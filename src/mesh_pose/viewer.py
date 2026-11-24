"""Tk operator display; no camera/GPU ownership and no pose state machine."""

from time import monotonic
import itertools
import json
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


def preview_lut(gamma: float) -> np.ndarray:
    """Display transfer with fixed black/white points and identity at gamma=1."""
    return np.rint(255 * (np.arange(256) / 255) ** (1 / gamma)).astype(np.uint8)


def preview_rgb(rgb: np.ndarray, lut: np.ndarray, denoise: bool) -> np.ndarray:
    # Filter one exposure at a time, before gamma amplifies shadow noise.
    if denoise:
        rgb = cv2.bilateralFilter(rgb, 5, 20, 2)
    return cv2.LUT(rgb, lut)


def annotate_pose(rgb, K, pose, bounds):
    canvas = rgb.copy()
    if pose is None:
        return canvas
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
                        cv2.line(canvas, tuple(uv[i]), tuple(uv[j]), (95, 211, 204), 1, cv2.LINE_AA)
    length = .045 if bounds is None else .65 * np.max(np.ptp(bounds, axis=0))
    points = np.vstack([np.zeros(3), np.eye(3) * length]) @ pose[:3, :3].T + pose[:3, 3]
    if np.all(points[:, 2] > 0):
        uv = points @ K.T
        uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
        _axes(canvas, uv[0], uv[1:])
    origin = np.array([48, canvas.shape[0] - 48])
    cv2.circle(canvas, tuple(origin), 42, (22, 31, 42), -1, cv2.LINE_AA)
    _axes(canvas, origin, origin + np.rint(29 * pose[:2, :3].T).astype(int))
    return canvas


def _axes(image, origin, ends):
    for axis, end, color in zip("XYZ", ends, [(255, 91, 97), (106, 235, 131), (80, 163, 255)]):
        cv2.line(image, tuple(origin), tuple(end), (18, 23, 30), 5, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(origin), tuple(end), color, 2, cv2.LINE_AA, tipLength=.22)
        pos = (int(np.clip(end[0] + 3, 2, image.shape[1] - 16)),
               int(np.clip(end[1] - 3, 16, image.shape[0] - 4)))
        for ink, width in [((13, 19, 27), 4), (color, 1)]:
            cv2.putText(image, axis, pos, cv2.FONT_HERSHEY_DUPLEX, .55, ink, width, cv2.LINE_AA)
    cv2.circle(image, tuple(origin), 3, (248, 250, 252), -1, cv2.LINE_AA)


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
        "quality_sequence": result.quality_sequence if result else None,
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
        self.root.configure(bg="#111923")
        header = tk.Frame(self.root, bg="#111923")
        header.pack(fill="x", padx=18, pady=10)
        tk.Label(header, text="物体位姿跟踪", bg="#111923", fg="#eff4fa",
                 font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        self.badge = tk.Label(header, padx=12, pady=3, font=("Microsoft YaHei UI", 10, "bold"))
        self.badge.pack(side="left", padx=14)
        tk.Button(header, text="停止   Esc", command=session.request_stop, relief="flat",
                  bg="#344556", fg="white", padx=12).pack(side="right")
        tone = tk.Frame(header, bg="#111923")
        tone.pack(side="right", padx=20)
        tk.Label(tone, text="预览亮度", bg="#111923", fg="#d0dce8",
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.preview_gamma = tk.DoubleVar(value=1.0)
        tk.Scale(tone, from_=1.0, to=2.5, resolution=.1, variable=self.preview_gamma,
                 orient="horizontal", length=120, showvalue=False, highlightthickness=0,
                 bg="#111923", troughcolor="#344556", bd=0).pack(side="left", padx=8)
        self.denoise = tk.BooleanVar(value=True)
        tk.Checkbutton(tone, text="降噪", variable=self.denoise, bg="#111923", fg="#d0dce8",
                       activebackground="#111923", activeforeground="white", selectcolor="#344556",
                       highlightthickness=0).pack(side="left")
        self._preview_cache = {}
        self._gamma = None
        self.picture = tk.Label(self.root, bg="#111923", borderwidth=0)
        self.picture.pack(fill="x")
        footer = tk.Frame(self.root, bg="#1c2937")
        footer.pack(fill="x")
        self.xyz = []
        for axis, color in zip("XYZ", ("#ff8990", "#86df9c", "#80b6ff")):
            card = tk.Frame(footer, bg="#1c2937")
            card.pack(side="left", padx=18, pady=9)
            tk.Label(card, text=f"{axis}   /   mm", bg="#1c2937", fg=color,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w")
            value = tk.Label(card, width=8, anchor="w", bg="#1c2937", fg="white", font=("Consolas", 16))
            value.pack()
            self.xyz.append(value)
        self.status = tk.Label(footer, bg="#1c2937", justify="right", font=("Microsoft YaHei UI", 10))
        self.status.pack(side="right", padx=16)
        self.rotation = tk.Label(self.root, bg="#111923", fg="#b5c5d5", font=("Consolas", 10), anchor="w")
        self.rotation.pack(fill="x", padx=16, pady=6)

    @staticmethod
    def _callback_error(exc_type, exc_value, traceback):
        # Tk otherwise prints callback failures and leaves a frozen UI running.
        raise exc_value.with_traceback(traceback)

    def _display_image(self, frame, settings, slot):
        key = (frame.sequence, settings)
        cached = self._preview_cache.get(slot)
        if cached is None or cached[0] != key:
            cached = key, preview_rgb(frame.rgb, self._display_lut, settings[1])
            self._preview_cache[slot] = cached
        return cached[1]

    def _tick(self):
        now = monotonic()
        if (self.duration_s > 0 and now - self.started >= self.duration_s) or (self.directory / "STOP").exists():
            self.session.request_stop()
        snapshot = self.session.snapshot()
        status = live_status(snapshot, now, self.max_pose_age_s, self.session.stop.is_set())
        frame, estimate = snapshot["frame"], snapshot["estimate"]
        if frame is not None:
            gamma = self.preview_gamma.get()
            if gamma != self._gamma:
                self._display_lut = preview_lut(gamma)
                self._gamma = gamma
            settings = (gamma, self.denoise.get())
            left = self._display_image(frame.frame, settings, "live")
            if estimate is not None:
                right_rgb = (left if estimate.source.frame.sequence == frame.frame.sequence else
                             self._display_image(estimate.source.frame, settings, "pose"))
                right = annotate_pose(right_rgb, estimate.source.frame.K,
                                      estimate.result.T_camera_object if status["valid_now"] else None,
                                      snapshot["bounds"])
            else:
                right = np.zeros_like(left)
            canvas = np.concatenate([left, right], axis=1)
            for x, title in [(0, "LIVE  /  RGB"), (left.shape[1], "POSE  /  RGB-D")]:
                cv2.rectangle(canvas, (x + 10, 10), (x + 158, 36), (24, 36, 49), -1)
                cv2.putText(canvas, title, (x + 18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            .49, (230, 241, 246), 1, cv2.LINE_AA)
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
        for i, label in enumerate(self.xyz):
            label.configure(text=f"{1000 * pose[i, 3]:.1f}" if pose is not None else "—")
        rows = ("    ".join("[" + "  ".join(f"{v:+.3f}" for v in row) + "]" for row in pose[:3, :3])
                if pose is not None else "—")
        self.rotation.configure(text=f"R    {rows}")
        valid = status["valid_now"]
        confirmation = "复核目标轮廓" if status["reason"] == "confirming_detection" else "确认新位姿"
        states = {"TRACKING": "跟踪中" if valid else "位姿已过期", "LOST": "重新定位",
                  "CONFIRMING": confirmation, "SEARCHING": "搜索目标", "STARTING": "启动中"}
        self.badge.configure(text=states.get(status["state"], status["state"]),
                             bg="#19443e" if valid else "#533f22", fg="#99efce" if valid else "#ffd898")
        rate = status["inference"].get("valid_pose_hz", 0)
        camera = status["capture"].get("received_fps", 0)
        latency = status["pose_age_s"]
        age = "—" if latency is None else f"{latency * 1000:.0f} ms"
        self.status.configure(text=status["error"] or
                              f"有效位姿   {rate:.1f} Hz     相机   {camera:.1f} Hz\n位姿帧龄   {age}     {status['reason']}",
                              fg="#ff8990" if status["error"] else "#d0dce8")
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
