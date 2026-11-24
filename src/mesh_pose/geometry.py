"""Independent rendered-depth verification; tolerances are not accuracy guarantees."""

from collections.abc import Callable

import cv2
import numpy as np

from .contracts import GeometryConfig, Quality, RGBDFrame


def touches_image_boundary(mask: np.ndarray) -> bool:
    """A clipped silhouette cannot establish a full-mesh registration."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def candidate_has_depth_support(frame: RGBDFrame, mask: np.ndarray,
                                diameter_m: float, config: GeometryConfig) -> bool:
    """Reject missing-depth, tiny-part and room-sized proposals before spending the top-K budget."""
    if mask.shape != frame.depth_m.shape:
        raise ValueError("Candidate mask dimensions do not match RGB-D")
    ys, xs = np.where(mask & (frame.depth_m > 0))
    if len(xs) < config.min_pixels or len(xs) / max(np.count_nonzero(mask), 1) < config.min_depth_coverage:
        return False
    depth = frame.depth_m[ys, xs]
    points = np.column_stack(((xs - frame.K[0, 2]) * depth / frame.K[0, 0],
                              (ys - frame.K[1, 2]) * depth / frame.K[1, 1], depth))
    # Robust bounds reduce the effect of isolated depth speckles; this is only a coarse gate.
    extent = float(np.linalg.norm(np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)))
    return config.min_candidate_extent_factor * diameter_m <= extent <= config.max_candidate_extent_factor * diameter_m


def is_rigid_pose(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    rotation = pose[:3, :3]
    return bool(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
                and abs(np.linalg.det(rotation) - 1) < 2e-3 and pose[2, 3] > 0)


def pose_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))))
    return distance, angle


def surface_center(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, min_pixels: int):
    """Median visible surface in camera coordinates (metres), not the CAD origin."""
    y, x = np.nonzero(mask & (depth > 0))
    if len(x) < min_pixels:
        return None
    z = depth[y, x]
    return np.array([np.median((x - K[0, 2]) * z / K[0, 0]),
                     np.median((y - K[1, 2]) * z / K[1, 1]), np.median(z)])


def refine_depth_pose(frame: RGBDFrame, initial_pose: np.ndarray,
                      render: Callable[[RGBDFrame, np.ndarray], np.ndarray],
                      config: GeometryConfig, mask: np.ndarray | None = None) -> np.ndarray:
    """Bounded projective point-to-plane ICP after learned refinement.

    Updates are about the object center. Unobservable directions retain their prior;
    no candidate outside the local trust region is accepted. The full verifier still runs.
    """
    if not is_rigid_pose(initial_pose):
        return initial_pose
    pose = initial_pose.copy()
    best = pose.copy()
    rendered_depth = render(frame, pose)
    best_quality = verify_depth(frame.depth_m, rendered_depth, config, mask)
    height, width = frame.depth_m.shape
    for _ in range(config.icp_iterations):
        footprint = rendered_depth > 0
        if np.count_nonzero(footprint) < config.min_pixels:
            break
        # Normals and linear algebra only need the object footprint. One-pixel
        # padding preserves the exact neighbor stencil used on the full image.
        x, y, w, h = cv2.boundingRect(footprint.view(np.uint8))
        region = (slice(max(0, y - 1), min(height, y + h + 1)),
                  slice(max(0, x - 1), min(width, x + w + 1)))
        depth = rendered_depth[region]
        observed_depth = frame.depth_m[region]
        yy, xx = np.mgrid[region]
        rays = np.stack(((xx - frame.K[0, 2]) / frame.K[0, 0],
                         (yy - frame.K[1, 2]) / frame.K[1, 1], np.ones_like(xx)), axis=-1)
        points = depth[..., None] * rays
        observed = observed_depth[..., None] * rays
        normals = np.cross(points[1:-1, 2:] - points[1:-1, :-2],
                           points[2:, 1:-1] - points[:-2, 1:-1])
        lengths = np.linalg.norm(normals, axis=-1)
        normals /= np.maximum(lengths[..., None], 1e-12)
        support = (depth > 0) & (observed_depth > 0)
        support &= np.abs(depth - observed_depth) < config.icp_max_correspondence_m
        if mask is not None:
            support &= mask[region]
        support = cv2.erode(support.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)[1:-1, 1:-1]
        support &= lengths > 1e-10
        if np.count_nonzero(support) < config.min_pixels:
            break
        p = points[1:-1, 1:-1][support]
        q = observed[1:-1, 1:-1][support]
        n = normals[support]
        residual = np.sum((p - q) * n, axis=1)
        keep = np.abs(residual) <= np.quantile(np.abs(residual), 0.8)
        p, n, residual = p[keep], n[keep], residual[keep]
        if len(residual) < config.min_pixels:
            break
        jacobian = np.column_stack((n, np.cross(p - pose[:3, 3], n)))
        delta = np.linalg.lstsq(jacobian, -residual, rcond=1e-3)[0]
        if not np.isfinite(delta).all():
            break
        scale = max(1.0, np.linalg.norm(delta[:3]) / 0.005,
                    np.linalg.norm(delta[3:]) / np.deg2rad(3.0))
        delta /= scale
        candidate = pose.copy()
        candidate[:3, :3] = cv2.Rodrigues(delta[3:])[0] @ pose[:3, :3]
        candidate[:3, 3] += delta[:3]
        distance, angle = pose_difference(initial_pose, candidate)
        if distance > config.icp_max_translation_m or angle > config.icp_max_rotation_deg:
            break
        rendered_depth = render(frame, candidate)
        quality = verify_depth(frame.depth_m, rendered_depth, config, mask)
        # Never trade an accepted pose for one rejected by the full geometric gate.
        candidate_rank = (quality.rank, -quality.median_error_m if quality.median_error_m is not None else -np.inf)
        best_rank = (best_quality.rank, -best_quality.median_error_m if best_quality.median_error_m is not None else -np.inf)
        if candidate_rank > best_rank and (quality.accepted or not best_quality.accepted):
            best, best_quality = candidate.copy(), quality
        pose = candidate
        if np.linalg.norm(delta[:3]) < 1e-5 and np.linalg.norm(delta[3:]) < 1e-4:
            break
    return best


def verify_depth(observed: np.ndarray, rendered: np.ndarray, config: GeometryConfig,
                 mask: np.ndarray | None = None) -> Quality:
    if rendered.shape != observed.shape or not np.isfinite(rendered).all():
        return Quality(False, "invalid_render")
    silhouette = rendered > 0
    pixels = int(np.count_nonzero(silhouette))
    if pixels < config.min_pixels:
        return Quality(False, "insufficient_projected_area", rendered_pixels=pixels)
    # Only projected pixels enter the residual statistics. For IoU the
    # union still includes mask pixels outside the projected footprint.
    measured = observed[silhouette]
    has_depth = measured > 0
    residual = measured - rendered[silhouette]
    occluded = has_depth & (residual < -config.occlusion_tolerance_m)
    visible = has_depth & ~occluded
    residual = residual[visible]
    count = len(residual)
    coverage = np.count_nonzero(has_depth) / pixels
    fraction = count / pixels
    absolute = np.abs(residual)
    inlier = np.count_nonzero(absolute <= config.depth_tolerance_m) / max(count, 1)
    free_space = np.count_nonzero(residual > config.depth_tolerance_m) / max(count, 1)
    error = float(np.median(absolute)) if count else None
    iou = None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError("Candidate mask dimensions do not match RGB-D")
        intersection = np.count_nonzero(mask[silhouette] & ~occluded)
        union = np.count_nonzero(mask) + pixels - np.count_nonzero(occluded) - intersection
        iou = intersection / max(union, 1)
    reason = "ok"
    for failed, label in (
        (coverage < config.min_depth_coverage, "insufficient_depth"),
        (count < config.min_pixels or fraction < config.min_visible_fraction, "insufficient_visibility"),
        (free_space > config.max_free_space_ratio, "free_space_conflict"),
        (inlier < config.min_inlier_ratio, "depth_mismatch"),
        (iou is not None and iou < config.min_mask_iou, "mask_mismatch"),
    ):
        if failed:
            reason = label
            break
    return Quality(reason == "ok", reason, pixels, coverage, fraction, inlier, free_space, error, iou)

