"""Fit and verify a CAD pose against one RGB-D observation."""

from dataclasses import replace
import logging

import cv2
import numpy as np

from .contracts import Candidate, Config, PoseBackend, Quality, RGBDFrame
from .geometry import is_rigid_pose, pose_difference, surface_center, touches_image_boundary, verify_depth

logger = logging.getLogger(__name__)


class PoseMatcher:
    def __init__(self, backend: PoseBackend, config: Config):
        self.backend = backend
        self.config = config

    def within_step(self, seed, pose) -> bool:
        distance, angle = pose_difference(seed, pose)
        limits = self.config.tracking
        return distance <= limits.max_translation_step_m and angle <= limits.max_rotation_step_deg

    def quality(self, frame: RGBDFrame, pose: np.ndarray, mask=None, rendered=None) -> Quality:
        if not is_rigid_pose(pose):
            return Quality(False, "invalid_pose")
        if rendered is None:
            rendered = self.backend.render_depth(frame, pose)
        return verify_depth(frame.depth_m, rendered, self.config.geometry, mask)

    def registration_quality(self, frame: RGBDFrame, pose: np.ndarray, mask, rendered=None) -> Quality:
        if rendered is None and is_rigid_pose(pose):
            rendered = self.backend.render_depth(frame, pose)
        quality = self.quality(frame, pose, mask, rendered)
        if quality.accepted and touches_image_boundary(rendered > 0):
            return replace(quality, accepted=False, reason="truncated_registration")
        if quality.accepted and (
            quality.inlier_ratio < self.config.geometry.min_registration_inlier_ratio
            or quality.mask_iou < self.config.geometry.min_registration_mask_iou
        ):
            return replace(quality, accepted=False, reason="weak_registration_geometry")
        return quality

    def mask_quality(self, frame, pose, rendered, *, strong):
        ys, xs = np.nonzero(rendered > 0)
        if not len(xs):
            return None
        pad = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) // 2 + 1
        height, width = frame.depth_m.shape
        region = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                  min(width, int(xs.max()) + pad + 1), min(height, int(ys.max()) + pad + 1))
        check = self.registration_quality if strong else self.quality
        def verify(image, area):
            qualities = (check(frame, pose, c.mask, rendered)
                         for c in self.backend.detect(image, region=area)
                         if c.similarity >= self.config.backend.min_detection_similarity)
            return max((q for q in qualities if q.accepted), key=lambda q: q.rank, default=None)

        accepted = verify(frame, region)
        if accepted is None and not strong:
            # A dark crop can split one object into separate faces. Only retry
            # its segmentation; all metric checks use the original exposure.
            x0, y0, x1, y1 = region
            gray = cv2.cvtColor(frame.rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            median = float(np.median(gray)) / 255
            if 0 < median < .45:
                gamma = min(2.5, np.log(median) / np.log(.45))
                lut = np.rint(255 * (np.arange(256) / 255) ** (1 / gamma)).astype(np.uint8)
                accepted = verify(replace(frame, rgb=cv2.LUT(frame.rgb, lut)), region)
        if accepted is None and strong and region != (0, 0, width, height):
            accepted = verify(frame, None)
            logger.info("Full-image initialization verification: accepted=%s", accepted is not None)
        return accepted

    def verify_registration_candidates(self, frame: RGBDFrame, pose: np.ndarray,
                                         candidates: list[Candidate], rendered=None) -> Quality:
        """The seed mask proposes a pose; any qualified full silhouette can verify it."""
        if not is_rigid_pose(pose):
            return Quality(False, "invalid_pose")
        if rendered is None:
            rendered = self.backend.render_depth(frame, pose)
        qualities = [self.registration_quality(frame, pose, c.mask, rendered) for c in candidates
                     if c.similarity >= self.config.backend.min_detection_similarity]
        return max(qualities, key=lambda q: (q.accepted, q.rank),
                   default=Quality(False, "no_detection"))

    def local_fit(self, frame: RGBDFrame, last_pose, previous_render=None):
        if last_pose is None:
            return None
        # The retained pose is a private hint, never a published measurement.
        # Do not refine onto a foreground occluder when the old footprint is hidden.
        previous_quality = self.quality(frame, last_pose, rendered=previous_render)
        if previous_quality.reason in {"insufficient_visibility", "insufficient_depth"}:
            return None
        pose = self.backend.track(frame, last_pose.copy())
        if not is_rigid_pose(pose):
            return None
        if not self.within_step(last_pose, pose):
            return None
        rendered = self.backend.render_depth(frame, pose)
        if not self.quality(frame, pose, rendered=rendered).accepted:
            return None
        return pose, rendered

    def surface_relocation(self, frame, seed, rendered, candidates):
        # Align *visible surfaces*: their offsets from the mesh origin generally
        # do not vanish. Use the shift only as an initializer on this RGB-D frame.
        minimum = self.config.geometry.min_pixels
        reference = surface_center(rendered, rendered > 0, frame.K, minimum)
        if reference is None:
            return None
        matches = []
        for c in candidates[:self.config.backend.max_candidates]:
            if c.similarity < self.config.backend.min_detection_similarity:
                continue
            observed = surface_center(frame.depth_m, c.mask, frame.K, minimum)
            if observed is None:
                continue
            shift = observed - reference
            if np.linalg.norm(shift) > self.config.tracking.max_translation_step_m:
                continue
            guess = seed.copy()
            guess[:3, 3] += shift
            pose = self.backend.track(frame, guess)
            if not is_rigid_pose(pose):
                continue
            if not self.within_step(seed, pose):
                continue
            quality = self.registration_quality(frame, pose, c.mask)
            if quality.accepted:
                matches.append((pose, quality))
        return self.unique_match(matches)

    def unique_match(self, matches):
        if not matches:
            return None
        best, quality = max(matches, key=lambda item: item[1].rank)
        if any(pose_difference(best, other)[0] > self.backend.diameter_m * .35 for other, _ in matches):
            return None
        return best, quality

    def refresh_initialization(self, frame: RGBDFrame, seed: np.ndarray,
                                candidates: list[Candidate] | None = None, *, rotation_budget: int | None = None) -> tuple[np.ndarray, Quality] | None:
        """A globally verified but old seed may need more than small-step tracking."""
        steps = self.refresh_steps(frame, seed, candidates, rotation_budget=rotation_budget)
        try:
            while True:
                next(steps)
        except StopIteration as finished:
            return finished.value

    def refresh_steps(self, frame, seed, candidates, *, rotation_budget=None):
        rendered = self.backend.render_depth(frame, seed)
        silhouette = rendered > 0
        matches = []
        if candidates is None:
            candidates = self.backend.detect(frame)
        # Use prior spatial support to prioritize masks without asking the user
        # to click. Final pose, motion and strong mask/depth gates remain mandatory.
        candidates = sorted(candidates, key=lambda c: np.count_nonzero(c.mask & silhouette)
                        / max(np.count_nonzero(c.mask | silhouette), 1), reverse=True)
        relocation = self.surface_relocation(frame, seed, rendered, candidates)
        if relocation is not None:
            logger.info("Surface-aligned seed verified on frame %d", frame.sequence)
            return relocation
        eligible = [c for c in candidates[:self.config.backend.max_candidates]
                    if c.similarity >= self.config.backend.min_detection_similarity
                    and np.any(c.mask & silhouette)]
        # During a live tracking failure, search at most one unambiguous region
        # on this exposure. Leave competing regions to cooperative global search.
        if rotation_budget is not None and len(eligible) > rotation_budget:
            return None
        for candidate in eligible:
            yield  # Allow fresh-frame recovery before starting another rotation search.
            pose = self.backend.relocalize_seed(
                frame, candidate.mask, seed.copy(), self.config.tracking.max_rotation_step_deg)
            if not is_rigid_pose(pose):
                continue
            if not self.within_step(seed, pose):
                continue
            quality = self.verify_registration_candidates(frame, pose, candidates)
            logger.info("Fresh initialization: accepted=%s reason=%s inliers=%.3f mask_iou=%s",
                        quality.accepted, quality.reason, quality.inlier_ratio, quality.mask_iou)
            if quality.accepted:
                matches.append((pose, quality))
        return self.unique_match(matches)

    def retry_from_seed(self, frame, seed):
        if seed is None:
            return None
        pose = self.backend.track(frame, seed)
        if not is_rigid_pose(pose):
            return None
        rendered = self.backend.render_depth(frame, pose)
        if not self.quality(frame, pose, rendered=rendered).accepted:
            return None
        quality = self.mask_quality(frame, pose, rendered, strong=True)
        return (pose, quality) if quality is not None else None

    def register_scene(self, frame, candidates, *, cooperative, full_retry):
        """Return an unpublished pose on this observation after the full-grid checks."""
        hypotheses = []
        last_quality = None
        nominated = []
        attempted = False
        proposals = () if full_retry else self.backend.guided_poses(frame, candidates)
        for proposal in proposals:
            if proposal is not None:
                attempted = True
                index, pose = proposal
                if self.verify_registration_candidates(frame, pose, candidates).accepted:
                    nominated.append(index)
            yield
        if attempted:
            logger.info("Geometry shortlist selected seed indices %s", nominated)
        if cooperative and attempted and not nominated:
            # After a miss, spend the full-grid budget on the next live frame.
            return "retry_full_on_fresh_frame", None, None
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
                quality = self.verify_registration_candidates(frame, pose, candidates)
                logger.info("Registration: similarity=%.3f accepted=%s reason=%s inliers=%.3f mask_iou=%s",
                            candidate.similarity, quality.accepted, quality.reason,
                            quality.inlier_ratio, quality.mask_iou)
                last_quality = quality
                if quality.accepted:
                    hypotheses.append((pose.copy(), quality))
            if hypotheses:
                break
        if not hypotheses:
            return "no_verified_candidate", last_quality, None
        match = self.unique_match(hypotheses)
        if match is None:
            return "ambiguous_instances", max((q for _, q in hypotheses), key=lambda q: q.rank), None
        pose, quality = match
        return "confirming", quality, pose
