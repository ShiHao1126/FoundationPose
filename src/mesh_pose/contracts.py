"""Input/output contracts and validated shared settings. All geometry uses meters."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from collections.abc import Generator
import json

import numpy as np


@dataclass(frozen=True)
class BackendConfig:
    template_views: int
    descriptor_batch_size: int
    max_candidates: int
    min_detection_similarity: float
    register_iterations: int
    track_iterations: int


@dataclass(frozen=True)
class GeometryConfig:
    depth_tolerance_m: float
    occlusion_tolerance_m: float
    min_pixels: int
    min_depth_coverage: float
    min_visible_fraction: float
    min_inlier_ratio: float
    max_free_space_ratio: float
    min_mask_iou: float
    min_registration_inlier_ratio: float
    min_registration_mask_iou: float
    min_candidate_extent_factor: float
    max_candidate_extent_factor: float
    icp_iterations: int
    icp_max_correspondence_m: float
    icp_max_translation_m: float
    icp_max_rotation_deg: float


@dataclass(frozen=True)
class TrackingConfig:
    confirmation_frames: int
    search_interval_frames: int
    detection_check_interval_frames: int
    max_frame_gap_s: float
    max_translation_step_m: float
    max_rotation_step_deg: float


@dataclass(frozen=True)
class Config:
    backend: BackendConfig
    geometry: GeometryConfig
    tracking: TrackingConfig

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"backend", "geometry", "tracking"}:
            raise ValueError("Config must contain exactly backend, geometry, tracking")
        config = cls(BackendConfig(**raw["backend"]), GeometryConfig(**raw["geometry"]),
                     TrackingConfig(**raw["tracking"]))
        for group in (config.backend, config.geometry, config.tracking):
            for name, value in asdict(group).items():
                if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                if group.__annotations__[name] is int and not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")
                if name.endswith(("_ratio", "_fraction", "_coverage", "_iou", "_similarity")) and value > 1:
                    raise ValueError(f"{name} must be <= 1")
        if config.geometry.occlusion_tolerance_m < config.geometry.depth_tolerance_m:
            raise ValueError("occlusion_tolerance_m must be >= depth_tolerance_m")
        if config.geometry.min_candidate_extent_factor >= config.geometry.max_candidate_extent_factor:
            raise ValueError("Candidate extent bounds must be increasing")
        return config


@dataclass(frozen=True)
class RGBDFrame:
    sequence: int
    timestamp_s: float
    rgb: np.ndarray  # uint8 H,W,3 RGB; rectified and registered to depth
    depth_m: np.ndarray  # float32 H,W camera Z, meters; zero = invalid
    K: np.ndarray  # rectified RGB pinhole intrinsics, 3x3

    def validate(self) -> None:
        if self.sequence < 0 or not np.isfinite(self.timestamp_s):
            raise ValueError("Invalid sequence or timestamp")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("RGB must be uint8 (H,W,3)")
        if self.depth_m.shape != self.rgb.shape[:2] or self.depth_m.dtype != np.float32:
            raise ValueError("Depth must be aligned float32 (H,W) in meters")
        if not np.isfinite(self.depth_m).all() or np.any(self.depth_m < 0):
            raise ValueError("Depth must be nonnegative and finite; use zero for missing depth")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("K must be finite 3x3")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        if abs(self.K[0, 1]) > 1e-6 or abs(self.K[1, 0]) > 1e-6:
            raise ValueError("Images must be rectified with zero-skew intrinsics")


@dataclass(frozen=True)
class Candidate:
    mask: np.ndarray
    similarity: float  # cosine similarity; not a calibrated probability


@dataclass(frozen=True)
class Quality:
    accepted: bool
    reason: str
    rendered_pixels: int = 0
    depth_coverage: float = 0.0
    visible_fraction: float = 0.0
    inlier_ratio: float = 0.0
    free_space_ratio: float = 1.0
    median_error_m: float | None = None
    mask_iou: float | None = None

    @property
    def rank(self) -> float:
        return self.inlier_ratio - self.free_space_ratio + (self.mask_iou or 0.0)


@dataclass(frozen=True)
class TrackingResult:
    sequence: int
    timestamp_s: float
    state: str
    reason: str
    T_camera_object: np.ndarray | None  # original CAD axes/origin, metric translation
    quality: Quality | None
    elapsed_s: float

    def to_record(self) -> dict:
        record = asdict(self)
        record.update(schema_version=1, valid=self.T_camera_object is not None,
                      translation_unit="m", camera_axes="x-right,y-down,z-forward",
                      object_frame="original_mesh_origin_and_axes")
        record["T_camera_object"] = None if self.T_camera_object is None else self.T_camera_object.tolist()
        return record


class PoseBackend(Protocol):
    """Implementations own model resources; Tracker owns accepted pose and lifecycle state."""

    diameter_m: float

    def detect(self, frame: RGBDFrame, *, region: tuple[int, int, int, int] | None = None) -> list[Candidate]: ...
    def guided_poses(self, frame: RGBDFrame, candidates: list[Candidate]) -> Generator[tuple[int, np.ndarray] | None, None, None]: ...
    def register(self, frame: RGBDFrame, mask: np.ndarray) -> np.ndarray: ...
    def register_steps(self, frame: RGBDFrame, mask: np.ndarray) -> Generator[None, None, np.ndarray]: ...
    def relocalize_seed(self, frame: RGBDFrame, mask: np.ndarray, seed: np.ndarray,
                       max_rotation_deg: float) -> np.ndarray: ...
    def track(self, frame: RGBDFrame, previous_pose: np.ndarray) -> np.ndarray: ...
    def render_depth(self, frame: RGBDFrame, pose: np.ndarray) -> np.ndarray: ...
    def to_object_frame(self, pose: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...

