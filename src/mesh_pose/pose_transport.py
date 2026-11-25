"""Transport a private CAD pose through past RGB-D observations after alignment fails."""

from time import perf_counter

import cv2
import numpy as np

from .geometry import is_rigid_pose


def _flow_pose(matcher, previous, current, pose):
    rendered = matcher.backend.render_depth(previous, pose)
    mask = ((rendered > 0) & (previous.depth_m > 0) &
            (np.abs(rendered - previous.depth_m) < matcher.config.geometry.depth_tolerance_m)).astype(np.uint8)
    if cv2.countNonZero(mask) < matcher.config.geometry.min_pixels:
        return None
    x, y, width, height = cv2.boundingRect(mask)
    pad = max(24, width // 2, height // 2)
    left, top = max(0, x - pad), max(0, y - pad)
    area = np.s_[top:y+height+pad, left:x+width+pad]
    before = cv2.cvtColor(previous.rgb[area], cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(current.rgb[area], cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(before, 90, 0.006, 3, mask=mask[area], blockSize=3)
    if points is None or len(points) < 8:
        return None
    options = dict(winSize=(17, 17), maxLevel=3)
    forward, valid_forward, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **options)
    if forward is None:
        return None
    backward, valid_backward, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, **options)
    if backward is None:
        return None
    keep = (valid_forward[:, 0] & valid_backward[:, 0]).astype(bool)
    keep &= np.linalg.norm(backward[:, 0] - points[:, 0], axis=1) < 1.0
    if np.count_nonzero(keep) < 8:
        return None
    pixels = points[keep, 0] + [left, top]
    observed = forward[keep, 0] + [left, top]
    u, v = np.rint(pixels).astype(int).T
    z = rendered[v, u]
    K = previous.K
    camera_points = np.column_stack(((pixels[:, 0] - K[0, 2]) * z / K[0, 0],
                                     (pixels[:, 1] - K[1, 2]) * z / K[1, 1], z))
    object_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        object_points, observed, current.K, None, cv2.Rodrigues(pose[:3, :3])[0].astype(float),
        pose[:3, 3:].astype(float), True, iterationsCount=60, reprojectionError=2.,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or inliers is None or len(inliers) < max(6, len(observed) / 2):
        return None
    candidate = pose.copy()
    candidate[:3, :3] = cv2.Rodrigues(rotation)[0]
    candidate[:3, 3] = translation[:, 0]
    return candidate if is_rigid_pose(candidate) and matcher.within_step(pose, candidate) else None


def _correct(matcher, frame, seed):
    pose = matcher.backend.track(frame, seed.copy())
    if not is_rigid_pose(pose) or not matcher.within_step(seed, pose):
        return None
    rendered = matcher.backend.render_depth(frame, pose)
    if matcher.quality(frame, pose, rendered=rendered).accepted:
        return pose, rendered
    return None


def transport_pose(matcher, source, observations, seed):
    if not observations:
        return None
    deadline = perf_counter() + 3.5
    previous = source
    last_correction = source.timestamp_s
    pose = seed
    for frame in observations:
        if perf_counter() >= deadline:
            return None
        candidate = _flow_pose(matcher, previous, frame, pose)
        if (candidate is None or frame.timestamp_s - last_correction >= 0.5 or
                not matcher.quality(frame, candidate).accepted):
            fit = _correct(matcher, frame, pose if candidate is None else candidate)
            if fit is not None:
                candidate = fit[0]
                last_correction = frame.timestamp_s
            elif candidate is None:
                return None
        previous, pose = frame, candidate

    fit = _correct(matcher, previous, pose)
    if fit is None:
        return None
    pose, rendered = fit
    quality = matcher.mask_quality(previous, pose, rendered, strong=True)
    return (pose, quality) if quality is not None else None
