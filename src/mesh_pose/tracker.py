"""Single-object tracking, strong local recovery and cooperative global search."""

from time import perf_counter
from dataclasses import replace
import logging

import numpy as np

from .contracts import Candidate, Config, PoseBackend, Quality, RGBDFrame, TrackingResult
from .geometry import is_rigid_pose, pose_difference, touches_image_boundary, verify_depth

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, backend: PoseBackend, config: Config):
        self.backend = backend
        self.config = config
        self.state = "SEARCHING"
        self._pose = None
        self._confirmed = 0
        self._tracked = 0
        self._detection_pending = False
        self._frames_since_search = config.tracking.search_interval_frames
        self._previous_frame = None
        self._K = None
        self._last_good_pose = None
        self._search_steps = None
        self._retry_full = False
        self._strong_confirmation = False

    def _cancel_search(self) -> None:
        if self._search_steps is not None:
            self._search_steps.close()
            self._search_steps = None

    def _lose(self) -> None:
        self.state = "LOST"
        self._pose = None
        self._confirmed = 0
        self._tracked = 0
        self._detection_pending = False
        self._strong_confirmation = False
        self.backend.reset()

    def _quality(self, frame: RGBDFrame, pose: np.ndarray, mask=None, rendered=None) -> Quality:
        if not is_rigid_pose(pose):
            return Quality(False, "invalid_pose")
        if rendered is None:
            rendered = self.backend.render_depth(frame, pose)
        return verify_depth(frame.depth_m, rendered, self.config.geometry, mask)

    def _registration_quality(self, frame: RGBDFrame, pose: np.ndarray, mask, rendered=None) -> Quality:
        if rendered is None and is_rigid_pose(pose):
            rendered = self.backend.render_depth(frame, pose)
        quality = self._quality(frame, pose, mask, rendered)
        if quality.accepted and touches_image_boundary(rendered > 0):
            return replace(quality, accepted=False, reason="truncated_registration")
        if quality.accepted and (
            quality.inlier_ratio < self.config.geometry.min_registration_inlier_ratio
            or quality.mask_iou < self.config.geometry.min_registration_mask_iou
        ):
            return replace(quality, accepted=False, reason="weak_registration_geometry")
        return quality

    def _mask_quality(self, frame, pose, rendered, *, strong):
        ys, xs = np.nonzero(rendered > 0)
        if not len(xs):
            return None
        pad = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) // 2 + 1
        height, width = frame.depth_m.shape
        region = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                  min(width, int(xs.max()) + pad + 1), min(height, int(ys.max()) + pad + 1))
        check = self._registration_quality if strong else self._quality
        matched = [check(frame, pose, c.mask, rendered) for c in self.backend.detect(frame, region=region)
                   if c.similarity >= self.config.backend.min_detection_similarity]
        accepted = max((q for q in matched if q.accepted), key=lambda q: q.rank, default=None)
        if accepted is None and strong and region != (0, 0, width, height):
            # Cropping changes FastSAM's proposals. A missing local mask is not
            # evidence against a full-image mask passing the same strong checks.
            matched = [check(frame, pose, c.mask, rendered) for c in self.backend.detect(frame)
                       if c.similarity >= self.config.backend.min_detection_similarity]
            accepted = max((q for q in matched if q.accepted), key=lambda q: q.rank, default=None)
            logger.info("Full-image initialization verification: accepted=%s", accepted is not None)
        return accepted

    def _verify_registration_candidates(self, frame: RGBDFrame, pose: np.ndarray,
                                         candidates: list[Candidate]) -> Quality:
        """The seed mask proposes a pose; any qualified full silhouette can verify it."""
        if not is_rigid_pose(pose):
            return Quality(False, "invalid_pose")
        rendered = self.backend.render_depth(frame, pose)
        qualities = [self._registration_quality(frame, pose, c.mask, rendered) for c in candidates
                     if c.similarity >= self.config.backend.min_detection_similarity]
        return max(qualities, key=lambda q: (q.accepted, q.rank),
                   default=Quality(False, "no_detection"))

    def _recover_local(self, frame: RGBDFrame) -> Quality | None:
        if self._last_good_pose is None:
            return None
        # The retained pose is a private hint, never a published measurement.
        # Do not refine onto a foreground occluder when the old footprint is hidden.
        previous_quality = self._quality(frame, self._last_good_pose)
        if previous_quality.reason in {"insufficient_visibility", "insufficient_depth"}:
            return None
        pose = self.backend.track(frame, self._last_good_pose.copy())
        if not is_rigid_pose(pose):
            return None
        distance, angle = pose_difference(self._last_good_pose, pose)
        if distance > self.config.tracking.max_translation_step_m or angle > self.config.tracking.max_rotation_step_deg:
            return None
        rendered = self.backend.render_depth(frame, pose)
        if not self._quality(frame, pose, rendered=rendered).accepted:
            return None
        quality = self._mask_quality(frame, pose, rendered, strong=True)
        if quality is not None:
            self._cancel_search()
            self._retry_full = False
            self._pose = pose.copy()
            self._confirmed = 1
            self._tracked = 0
            self._strong_confirmation = True
            self.state = "CONFIRMING"
        return quality

    def _refresh_initialization(self, frame: RGBDFrame, seed: np.ndarray) -> tuple[np.ndarray, Quality] | None:
        """A globally verified but old seed may need more than small-step tracking."""
        rendered = self.backend.render_depth(frame, seed)
        silhouette = rendered > 0
        candidates = self.backend.detect(frame)
        # Use prior spatial support to prioritize masks without asking the user
        # to click. Final pose, motion and strong mask/depth gates remain mandatory.
        candidates.sort(key=lambda c: np.count_nonzero(c.mask & silhouette)
                        / max(np.count_nonzero(c.mask | silhouette), 1), reverse=True)
        for candidate in candidates[:self.config.backend.max_candidates]:
            if candidate.similarity < self.config.backend.min_detection_similarity:
                continue
            if not np.any(candidate.mask & silhouette):
                continue
            pose = self.backend.relocalize_seed(
                frame, candidate.mask, seed.copy(), self.config.tracking.max_rotation_step_deg)
            if not is_rigid_pose(pose):
                continue
            distance, angle = pose_difference(seed, pose)
            if distance > self.config.tracking.max_translation_step_m or angle > self.config.tracking.max_rotation_step_deg:
                continue
            quality = self._verify_registration_candidates(frame, pose, candidates)
            logger.info("Fresh initialization: accepted=%s reason=%s inliers=%.3f mask_iou=%s",
                        quality.accepted, quality.reason, quality.inlier_ratio, quality.mask_iou)
            if quality.accepted:
                return pose, quality
        return None

    def _search(self, frame: RGBDFrame, cooperative: bool) -> tuple[str, Quality | None]:
        quality = self._recover_local(frame)
        if quality is not None:
            return "confirming_local_recovery", quality
        if self._search_steps is not None:
            return self._advance_search(cooperative)
        self._frames_since_search += 1
        if self._frames_since_search < self.config.tracking.search_interval_frames:
            return "waiting_to_search", None
        self._frames_since_search = 0
        self._search_steps = self._global_search(frame, cooperative)
        return self._advance_search(cooperative)

    def _advance_search(self, cooperative):
        try:
            while True:
                next(self._search_steps)
                if cooperative:
                    return "global_search_in_progress", None
        except StopIteration as finished:
            self._search_steps = None
            return finished.value

    def _global_search(self, frame: RGBDFrame, cooperative: bool):
        hypotheses = []
        last_quality = None
        candidates = self.backend.detect(frame)
        nominated = []
        full_retry = self._retry_full
        self._retry_full = False
        attempted = False
        proposals = () if full_retry else self.backend.guided_poses(frame, candidates)
        for proposal in proposals:
            if proposal is not None:
                attempted = True
                index, pose = proposal
                if self._verify_registration_candidates(frame, pose, candidates).accepted:
                    nominated.append(index)
            yield
        if attempted:
            logger.info("Geometry shortlist selected seed indices %s", nominated)
        if cooperative and attempted and not nominated:
            # After a miss, spend the full-grid budget on the next live frame.
            self._retry_full = True
            self._frames_since_search = self.config.tracking.search_interval_frames
            return "retry_full_on_fresh_frame", None
        # A partial rotation grid can pass depth checks at the wrong orientation.
        # Confirm shortlisted regions with the reference's full rotation grid.
        fallback = [i for i in range(min(len(candidates), self.config.backend.max_candidates))
                    if i not in nominated]
        for indices in (nominated, fallback):
            for index in indices:
                candidate = candidates[index]
                if candidate.similarity < self.config.backend.min_detection_similarity:
                    continue
                mask = np.asarray(candidate.mask, dtype=bool)
                if mask.shape != frame.depth_m.shape:
                    raise ValueError("Detector returned wrong mask shape")
                if np.count_nonzero(mask & (frame.depth_m > 0)) < self.config.geometry.min_pixels:
                    continue
                pose = yield from self.backend.register_steps(frame, mask)
                quality = self._verify_registration_candidates(frame, pose, candidates)
                logger.info("Registration: similarity=%.3f accepted=%s reason=%s inliers=%.3f mask_iou=%s",
                            candidate.similarity, quality.accepted, quality.reason,
                            quality.inlier_ratio, quality.mask_iou)
                last_quality = quality
                if quality.accepted:
                    hypotheses.append((quality.rank, pose.copy(), quality))
            if hypotheses:
                break
        if not hypotheses:
            self.backend.reset()
            return "no_verified_candidate", last_quality
        hypotheses.sort(key=lambda item: item[0], reverse=True)
        _, best, quality = hypotheses[0]
        # Initial registration uses stronger gates than continued tracking. If two
        # distinct instances still pass, mesh geometry alone cannot identify one.
        if any(pose_difference(best, pose)[0] > self.backend.diameter_m * 0.35
               for _, pose, _ in hypotheses[1:]):
            self.backend.reset()
            return "ambiguous_instances", quality
        self._pose = best
        self._confirmed = 1
        self._tracked = 0
        # This global search may span many live frames; always verify its seed
        # against a new frame before publication, even with confirmation_frames=1.
        self._strong_confirmation = True
        self.state = "CONFIRMING"
        return "confirming", quality

    def step(self, frame: RGBDFrame, *, cooperative: bool = False) -> TrackingResult:
        """Offline callers finish search; live callers yield between GPU batches."""
        start = perf_counter()
        try:
            fresh_confirmation = self._strong_confirmation
            frame.validate()
            if self._K is None:
                self._K = frame.K.copy()
            elif not np.allclose(self._K, frame.K, rtol=0, atol=1e-5):
                raise ValueError("Intrinsics changed; construct a new backend and tracker")
            if self._previous_frame is not None:
                previous_sequence, previous_time = self._previous_frame
                if frame.sequence <= previous_sequence or frame.timestamp_s <= previous_time:
                    raise ValueError("Frames must have strictly increasing sequence numbers and timestamps")
                if frame.timestamp_s - previous_time > self.config.tracking.max_frame_gap_s:
                    # A slow global registration is only an unpublished hypothesis.
                    # Refine it on a fresh frame and repeat the strong mask/depth
                    # checks; otherwise live input would endlessly re-register.
                    if self.state == "CONFIRMING":
                        fresh_confirmation = True
                    elif self._pose is not None:
                        self._lose()
                        self._frames_since_search = self.config.tracking.search_interval_frames
            self._previous_frame = (frame.sequence, frame.timestamp_s)
            quality = None
            if np.count_nonzero(frame.depth_m) < self.config.geometry.min_pixels:
                self._lose()
                self._frames_since_search = self.config.tracking.search_interval_frames
                reason = "missing_depth"
            elif self._pose is None:
                reason, quality = self._search(frame, cooperative)
            else:
                pose = self.backend.track(frame, self._pose.copy())
                quality = self._quality(frame, pose)
                reason = quality.reason
                if quality.accepted:
                    distance, angle = pose_difference(self._pose, pose)
                    if distance > self.config.tracking.max_translation_step_m or angle > self.config.tracking.max_rotation_step_deg:
                        reason = "motion_jump"
                    else:
                        self._tracked += 1
                        if fresh_confirmation or self._detection_pending or self._tracked % self.config.tracking.detection_check_interval_frames == 0:
                            rendered = self.backend.render_depth(frame, pose)
                            accepted = self._mask_quality(frame, pose, rendered, strong=fresh_confirmation)
                            if accepted is None:
                                if fresh_confirmation:
                                    reason = "fresh_confirmation_failed"
                                else:
                                    reason = "detection_check_failed" if self._detection_pending else "confirming_detection"
                            else:
                                quality = accepted
                if fresh_confirmation and reason not in {"ok", "confirming_detection"}:
                    logger.info("Initialization confirmation rejected: %s", reason)
                    refreshed = self._refresh_initialization(frame, self._pose)
                    if refreshed is not None:
                        pose, quality = refreshed
                        reason = "confirming_refreshed_initialization"
                if reason in {"confirming_detection", "confirming_refreshed_initialization"}:
                    # Both a detector miss and a newly relocated seed withhold
                    # output until a separate new frame passes confirmation.
                    self._pose = pose.copy()
                    self._confirmed = 1
                    self._detection_pending = reason == "confirming_detection"
                    self._strong_confirmation = reason == "confirming_refreshed_initialization"
                    self.state = "CONFIRMING"
                elif reason != "ok":
                    self._lose()
                    self._frames_since_search = self.config.tracking.search_interval_frames
                else:
                    self._detection_pending = False
                    self._pose = pose.copy()
                    self._confirmed += 1
                    self.state = "TRACKING" if self._confirmed >= self.config.tracking.confirmation_frames else "CONFIRMING"
                    reason = "tracked" if self.state == "TRACKING" else "confirming"
                    if self.state == "TRACKING":
                        self._strong_confirmation = False
            if self.state == "TRACKING":
                self._last_good_pose = self._pose.copy()
            exported = self.backend.to_object_frame(self._pose) if self.state == "TRACKING" else None
            return TrackingResult(frame.sequence, frame.timestamp_s, self.state, reason,
                                  exported, quality, perf_counter() - start)
        except Exception:
            # Backend faults stop the run; do not disguise faults as normal target loss.
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

