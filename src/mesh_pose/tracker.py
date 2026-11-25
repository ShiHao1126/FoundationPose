"""Single-object tracking, strong local recovery and cooperative global search."""

from time import perf_counter
from dataclasses import replace
from enum import Enum

import cv2
import numpy as np

from .contracts import Config, PoseBackend, Quality, RGBDFrame, TrackingResult
from .geometry import pose_difference
from .pose_matching import PoseMatcher
from .frame_history import FrameHistory
from .pose_transport import transport_pose


class Phase(Enum):
    SEARCHING = "SEARCHING"
    ALIGNING = "ALIGNING"  # Transport a registration seed to the current observation.
    VERIFYING = "VERIFYING"
    TRACKING = "TRACKING"
    RECHECKING = "RECHECKING"
    LOST = "LOST"


class Tracker:
    def __init__(self, backend: PoseBackend, config: Config, history: FrameHistory | None = None):
        self._history = history
        self._registration_frame = None
        self.backend = backend
        self.config = config
        self.matcher = PoseMatcher(backend, config)
        self._phase = Phase.SEARCHING
        self._pose = None
        self._confirmed = 0
        self._tracked = 0
        self._frames_since_search = config.tracking.search_interval_frames
        self._previous_frame = None
        self._motion_sample = None
        self._K = None
        self._last_good_pose = None
        self._search_steps = None
        self._search_source = None
        self._retry_full = False
        self._mask_probe_time = -np.inf

    @property
    def state(self) -> str:
        if self._phase in (Phase.ALIGNING, Phase.VERIFYING, Phase.RECHECKING):
            return "CONFIRMING"
        return self._phase.value

    def _cancel_search(self) -> None:
        self._registration_frame = None
        if self._search_steps is not None:
            self._search_steps.close()
            self._search_steps = None

    def _lose(self) -> None:
        self._phase = Phase.LOST
        self._frames_since_search = self.config.tracking.search_interval_frames
        self._pose = None
        self._confirmed = 0
        self._tracked = 0
        self.backend.reset()

    def _recover_local(self, frame: RGBDFrame) -> Quality | None:
        fit = self.matcher.local_fit(frame, self._last_good_pose)
        if fit is None:
            return None
        pose, rendered = fit
        quality = self.matcher.mask_quality(frame, pose, rendered, strong=True)
        if quality is not None:
            self._begin_recovery(pose)
        return quality

    def _probe_reappearance(self, frame):
        """Look around the last pose while an older registration is still running."""
        seed = self._last_good_pose
        if seed is None or frame.timestamp_s - self._mask_probe_time < self.config.tracking.recovery_probe_interval_s:
            return None
        self._mask_probe_time = frame.timestamp_s
        rendered = self.backend.render_depth(frame, seed)
        footprint = rendered > 0
        if not footprint.any():
            return None
        x, y, w, h = cv2.boundingRect(footprint.view(np.uint8))
        # Cover the permitted lateral displacement at the object's nearest depth.
        z = rendered[footprint].min()
        pad = np.ceil(frame.K[[0, 1], [0, 1]] * self.config.tracking.max_translation_step_m / z).astype(int)
        region = (max(0, x - pad[0]), max(0, y - pad[1]),
                  min(frame.rgb.shape[1], x + w + pad[0]), min(frame.rgb.shape[0], y + h + pad[1]))
        candidates = self.backend.detect(frame, region=region)
        # Both local hypotheses share one detection on this new frame. Repeating
        # full-image segmentation between every GPU batch stalls global recovery.
        if not candidates:
            return None
        fit = self.matcher.local_fit(frame, seed, previous_render=rendered)
        if fit is not None:
            pose, fitted_depth = fit
            quality = self.matcher.verify_registration_candidates(frame, pose, candidates, fitted_depth)
            if quality.accepted:
                return pose, quality
        return self.matcher.surface_relocation(frame, seed, rendered, candidates)

    def _begin_recovery(self, pose: np.ndarray, phase: Phase = Phase.VERIFYING) -> None:
        self._pose = pose.copy()
        self._confirmed = 1
        self._tracked = 0
        self._phase = phase

    def _search(self, frame: RGBDFrame, cooperative: bool):
        if cooperative and self._search_steps is not None:
            reason, quality, source = self._advance_search(True, recovery_slice=self._last_good_pose is not None)
            if self._search_steps is not None:
                match = self._probe_reappearance(frame)
                if match is not None:
                    pose, quality = match
                    self._begin_recovery(pose)
                    return "confirming_mask_recovery", quality, frame.sequence
            return reason, quality, source
        quality = self._recover_local(frame)
        if quality is not None:
            return "confirming_local_recovery", quality, frame.sequence
        if self._search_steps is not None:
            return self._advance_search(cooperative)
        self._frames_since_search += 1
        if self._frames_since_search < self.config.tracking.search_interval_frames:
            return "waiting_to_search", None, None
        self._frames_since_search = 0
        self._search_source = frame.sequence
        self._search_steps = self._global_search(frame, cooperative)
        return self._advance_search(cooperative)

    def _advance_search(self, cooperative, *, recovery_slice=False):
        # Probe the newest image between short groups of GPU work during loss.
        # A single upstream batch can run longer than this scheduling budget.
        deadline = perf_counter() + (0.15 if recovery_slice else 0.0)
        try:
            while True:
                next(self._search_steps)
                if cooperative and perf_counter() >= deadline:
                    return "global_search_in_progress", None, None
        except StopIteration as finished:
            self._search_steps = None
            reason, quality, pose = finished.value
            if pose is not None:
                self._begin_recovery(pose, Phase.ALIGNING)
            elif reason == "retry_full_on_fresh_frame":
                self._retry_full = True
                self._frames_since_search = self.config.tracking.search_interval_frames
            else:
                self.backend.reset()
            return reason, quality, self._search_source

    def _global_search(self, frame: RGBDFrame, cooperative: bool):
        self._registration_frame = frame if self._history is not None else None
        candidates = self.backend.detect(frame)
        if self._last_good_pose is not None and candidates:
            self._mask_probe_time = frame.timestamp_s
            refreshed = yield from self.matcher.refresh_steps(frame, self._last_good_pose, candidates)
            if refreshed is not None:
                pose, quality = refreshed
                return "confirming_mask_recovery", quality, pose
        full_retry = self._retry_full
        self._retry_full = False
        return (yield from self.matcher.register_scene(frame, candidates, cooperative=cooperative,
                                                      full_retry=full_retry))

    def _motion_seed(self, frame, previous_time):
        if self._motion_sample is None or previous_time is None:
            return None
        earlier_time, earlier = self._motion_sample
        ratio = min(1.5, (frame.timestamp_s - previous_time) / (previous_time - earlier_time))
        seed = self._pose.copy()
        seed[:3, 3] += ratio * (self._pose[:3, 3] - earlier[:3, 3])
        rotvec = cv2.Rodrigues(self._pose[:3, :3] @ earlier[:3, :3].T)[0]
        seed[:3, :3] = cv2.Rodrigues(rotvec * ratio)[0] @ self._pose[:3, :3]
        distance, angle = pose_difference(self._pose, seed)
        if distance > self.config.tracking.max_translation_step_m or angle > self.config.tracking.max_rotation_step_deg:
            return None
        return seed

    def _prefer_motion_seed(self, frame, seed):
        if seed is None:
            return False
        distance, angle = pose_difference(self._pose, seed)
        surface_travel = distance + self.backend.diameter_m * .5 * np.deg2rad(angle)
        if surface_travel < self.config.geometry.depth_tolerance_m:
            return False
        old = self.matcher.quality(frame, self._pose)
        predicted = self.matcher.quality(frame, seed)
        # Depth selects the starting point; the fitted measurement must still
        # pass the normal motion and geometry checks before publication.
        return predicted.accepted and (not old.accepted or predicted.rank > old.rank)

    def _observe(self, frame, previous_time):
        """Fit a current-frame measurement and apply the gates for the active phase."""
        prediction = self._motion_seed(frame, previous_time) if self._phase is Phase.TRACKING else None
        use_prediction = self._prefer_motion_seed(frame, prediction)
        seed = prediction if use_prediction else self._pose
        pose = self.backend.track(frame, seed.copy())
        quality = self.matcher.quality(frame, pose)
        if not quality.accepted and self._phase is Phase.TRACKING:
            alternate = self._pose if use_prediction else prediction
            if quality.reason in {"depth_mismatch", "free_space_conflict", "insufficient_visibility"}:
                retry = self.matcher.retry_from_seed(frame, alternate)
                if retry is not None:
                    pose, quality = retry
        if not quality.accepted:
            return pose, quality
        if not self.matcher.within_step(self._pose, pose):
            return pose, replace(quality, accepted=False, reason="motion_jump")
        check_due = (self._tracked + 1) % self.config.tracking.detection_check_interval_frames == 0
        if self._phase is Phase.TRACKING and not check_due:
            return pose, quality
        rendered = self.backend.render_depth(frame, pose)
        strong = self._phase in (Phase.ALIGNING, Phase.VERIFYING)
        verified = self.matcher.mask_quality(frame, pose, rendered, strong=strong)
        if verified is None:
            return pose, replace(quality, accepted=False, reason="fresh_confirmation_failed"
                                 if strong else "detection_check_failed")
        return pose, verified

    def _follow(self, frame, previous_time, cooperative):
        pose, quality = self._observe(frame, previous_time)
        if quality.accepted:
            self._pose = pose.copy()
            self._confirmed += 1
            self._tracked += 1
            if self._confirmed >= self.config.tracking.confirmation_frames:
                self._phase = Phase.TRACKING
                self._last_good_pose = self._pose.copy()
                self._cancel_search()
                self._retry_full = False
                return "tracked", quality
            self._phase = Phase.VERIFYING
            return "confirming", quality

        if self._phase is Phase.TRACKING and quality.reason == "detection_check_failed":
            self._phase = Phase.RECHECKING
            self._pose = pose.copy()
            self._confirmed = 1
            self._tracked += 1
            return "confirming_detection", quality

        # A new candidate, even after a previously confirmed track, always takes
        # the same fresh-frame verification path before it can be exported.
        failed_fit = quality.reason in {"depth_mismatch", "free_space_conflict", "insufficient_visibility"}
        if self._phase is Phase.ALIGNING or (self._phase is Phase.TRACKING and failed_fit):
            budget = None
            if cooperative and self._last_good_pose is not None:
                budget = 1 if self._phase is Phase.TRACKING else 0
            match = self.matcher.refresh_initialization(frame, self._pose, rotation_budget=budget)
            if match is not None:
                pose, quality = match
                self._begin_recovery(pose)
                return "confirming_refreshed_initialization", quality
        if self._phase is Phase.ALIGNING and self._registration_frame is not None:
            source = self._registration_frame
            self._registration_frame = None
            observations = self._history.between(source, frame)
            match = transport_pose(self.matcher, source, observations, self._pose)
            if match is not None:
                pose, quality = match
                self._begin_recovery(pose)
                return "confirming_flow_transport", quality
        self._lose()
        return quality.reason, quality

    def step(self, frame: RGBDFrame, *, cooperative: bool = False) -> TrackingResult:
        """Advance one observation. Only TRACKING can export its measured pose."""
        start = perf_counter()
        try:
            frame.validate()
            if self._K is None:
                self._K = frame.K.copy()
            elif not np.allclose(self._K, frame.K, rtol=0, atol=1e-5):
                raise ValueError("Intrinsics changed; construct a new backend and tracker")
            previous_time = None
            if self._previous_frame is not None:
                previous_sequence, previous_time = self._previous_frame
                if frame.sequence <= previous_sequence or frame.timestamp_s <= previous_time:
                    raise ValueError("Frames must have strictly increasing sequence numbers and timestamps")
                if frame.timestamp_s - previous_time > self.config.tracking.max_frame_gap_s:
                    if self._phase is Phase.RECHECKING:
                        self._phase = Phase.VERIFYING
                    elif self._phase is Phase.TRACKING:
                        self._lose()
            self._previous_frame = (frame.sequence, frame.timestamp_s)
            previous_pose = self._pose if self._phase is Phase.TRACKING else None
            source = frame.sequence
            if np.count_nonzero(frame.depth_m) < self.config.geometry.min_pixels:
                self._lose()
                reason, quality = "missing_depth", None
            elif self._phase in (Phase.SEARCHING, Phase.LOST):
                reason, quality, source = self._search(frame, cooperative)
            else:
                reason, quality = self._follow(frame, previous_time, cooperative)
            tracking = self._phase is Phase.TRACKING
            self._motion_sample = (previous_time, previous_pose) if previous_pose is not None and tracking else None
            exported = self.backend.to_object_frame(self._pose) if tracking else None
            return TrackingResult(frame.sequence, frame.timestamp_s, self.state, reason,
                                  exported, quality, perf_counter() - start,
                                  quality_sequence=source if quality is not None else None)
        except Exception:
            self._lose()
            self._cancel_search()
            raise

    def close(self) -> None:
        try:
            self._lose()
            self._cancel_search()
            self._last_good_pose = None
        finally:
            self.backend.close()

