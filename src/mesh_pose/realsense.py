"""One owner for RealSense streaming. Importing this module never opens a camera."""

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
import json

import numpy as np

from .contracts import RGBDFrame


@dataclass(frozen=True)
class RealSenseConfig:
    serial: str
    width: int
    height: int
    fps: int
    timeout_ms: int
    warmup_frames: int
    color_exposure: float | None = None  # SDK native units; RGB units differ from depth usec.
    color_gain: float | None = None

    @classmethod
    def load(cls, path: Path) -> "RealSenseConfig":
        settings = cls(**json.loads(path.read_text(encoding="utf-8")))
        if not settings.serial:
            raise ValueError("Select a RealSense serial explicitly")
        for name in ("width", "height", "fps", "timeout_ms", "warmup_frames"):
            value = getattr(settings, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (settings.color_exposure is None) != (settings.color_gain is None):
            raise ValueError("Set color_exposure and color_gain together, or both null to keep device settings")
        if settings.color_exposure is not None:
            if (not np.isfinite(settings.color_exposure) or settings.color_exposure <= 0
                    or not np.isfinite(settings.color_gain) or settings.color_gain < 0):
                raise ValueError("Invalid manual color exposure/gain")
        return settings


class RealSenseSource:
    def __init__(self, config: RealSenseConfig):
        self.config = config
        self._pipeline = None
        self._previous_number = None
        self._previous_timestamp = None
        self._timestamp_domain = None
        self.dropped_color_frames = 0
        self.duplicate_color_frames = 0
        self.last_frame_metadata = {}
        self._color_sensor = None
        self._saved_color_options = []

    def start(self) -> None:
        import pyrealsense2 as rs

        if self._pipeline is not None:
            raise RuntimeError("Camera already started")
        settings = rs.config()
        settings.enable_device(self.config.serial)
        settings.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        settings.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.rgb8, self.config.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(settings)
        self._pipeline = pipeline
        try:
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            if self.config.color_exposure is not None:
                self._color_sensor = color_sensor
                self._saved_color_options = [(option, color_sensor.get_option(option)) for option in (
                    rs.option.exposure, rs.option.gain, rs.option.enable_auto_exposure)]
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, self.config.color_exposure)
                color_sensor.set_option(rs.option.gain, self.config.color_gain)
            self.depth_scale_m = device.first_depth_sensor().get_depth_scale()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            if not np.allclose(intrinsics.coeffs, 0, atol=1e-9):
                raise ValueError("This camera requires RGB-D rectification before the pinhole pose backend")
            self.K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                               [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
            self.metadata = {
                "device": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
                "depth_scale_m": self.depth_scale_m,
                "width": self.config.width, "height": self.config.height, "requested_fps": self.config.fps,
                "intrinsics": self.K.tolist(), "distortion_model": str(intrinsics.model),
                "distortion_coefficients": intrinsics.coeffs, "alignment": "depth_to_color",
                "color_exposure_sdk_units": color_sensor.get_option(rs.option.exposure),
                "color_gain": color_sensor.get_option(rs.option.gain),
                "color_auto_exposure": bool(color_sensor.get_option(rs.option.enable_auto_exposure)),
            }
            self._align = rs.align(rs.stream.color)
            for _ in range(self.config.warmup_frames):
                pipeline.wait_for_frames(self.config.timeout_ms)
            self._previous_number = None
            self._previous_timestamp = None
            self._timestamp_domain = None
            self.dropped_color_frames = 0
            self.duplicate_color_frames = 0
        except Exception:
            self.close()
            raise

    def read(self) -> RGBDFrame:
        if self._pipeline is None:
            raise RuntimeError("Call start() before reading the camera")
        deadline = monotonic() + self.config.timeout_ms / 1000
        while True:
            remaining_ms = int((deadline - monotonic()) * 1000)
            if remaining_ms <= 0:
                raise TimeoutError("No new unique RGB-D frame before camera timeout")
            frames = self._pipeline.wait_for_frames(remaining_ms)
            color, native_depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not native_depth:
                continue
            number = color.get_frame_number()
            # D415 framesets can reuse a color exposure with a new depth exposure.
            # Never feed the same RGB timestamp twice into the tracking state machine.
            if self._previous_number is not None and number == self._previous_number:
                self.duplicate_color_frames += 1
                continue
            timestamp = color.get_timestamp() / 1000
            domain = str(color.get_frame_timestamp_domain())
            if self._previous_number is not None and (number < self._previous_number or timestamp <= self._previous_timestamp):
                raise RuntimeError("Camera frame number or timestamp went backwards")
            if self._timestamp_domain is not None and domain != self._timestamp_domain:
                raise RuntimeError("Camera timestamp domain changed; restart the source")
            aligned_depth = self._align.process(frames).get_depth_frame()
            rgb = np.asanyarray(color.get_data()).copy()
            raw = np.asanyarray(aligned_depth.get_data())
            depth = raw.astype(np.float32) * self.depth_scale_m
            frame = RGBDFrame(number, timestamp, rgb, depth, self.K.copy())
            frame.validate()
            if self._previous_number is not None:
                self.dropped_color_frames += number - self._previous_number - 1
            self._previous_number, self._previous_timestamp = number, timestamp
            self._timestamp_domain = domain
            depth_domain = str(native_depth.get_frame_timestamp_domain())
            self.last_frame_metadata = {
                "color_frame": number, "depth_frame": native_depth.get_frame_number(),
                "color_timestamp_ms": color.get_timestamp(), "depth_timestamp_ms": native_depth.get_timestamp(),
                "color_clock": domain, "depth_clock": depth_domain,
                "timestamp_skew_ms": color.get_timestamp() - native_depth.get_timestamp() if domain == depth_domain else None,
            }
            return frame

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        sensor, self._color_sensor = self._color_sensor, None
        saved, self._saved_color_options = self._saved_color_options, []
        restore_error = None
        try:
            for option, value in saved:
                try:
                    sensor.set_option(option, value)
                except Exception as exc:
                    restore_error = restore_error or exc
        finally:
            if pipeline is not None:
                pipeline.stop()
        if restore_error is not None:
            raise RuntimeError("Camera stopped but original color options could not all be restored") from restore_error

    def __enter__(self) -> "RealSenseSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

