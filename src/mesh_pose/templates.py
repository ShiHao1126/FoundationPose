"""Compact view-dependent geometry for ranking RGB-D registration seeds."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np


def save_cache(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as out:
        temporary = Path(out.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def crop_depth(depth, mask):
    y, x = np.nonzero(mask)
    x0, y0 = x.min(), y.min()
    width, height = x.max() - x0 + 1, y.max() - y0 + 1
    crop = np.s_[y0:y0 + height, x0:x0 + width]
    # Separate silhouette and valid depth: D415 holes still belong to the mask.
    def resize(a):
        return cv2.resize(a, (32, 32), interpolation=cv2.INTER_NEAREST)
    z = resize(np.where(mask[crop], depth[crop], 0))
    return resize(mask[crop].astype(np.uint8)).astype(bool), z, np.array([width, height])


class ViewBank:
    def __init__(self, depth, extent):
        self.depth = depth
        self.extent = extent
        self.mask = depth > 0

    @classmethod
    def build(cls, renderer, grid, diameter):
        distance = 6 * diameter
        focal = 96 * distance / (1.2 * diameter)
        K = np.array([[focal, 0, 47.5], [0, focal, 47.5], [0, 0, 1]], np.float32)
        depths, extents = [], []
        for rotation in grid:
            pose = rotation.copy()
            pose[:3, 3] = [0, 0, distance]
            z = renderer.render(K, (96, 96), pose)
            _, sample, extent = crop_depth(z, z > 0)
            center = np.median(sample[sample > 0])
            depths.append(np.where(sample > 0, sample - center + diameter, 0))
            extents.append(extent * center / focal)
        return cls(np.stack(depths), np.stack(extents))

    def match(self, frame, mask, diameter):
        silhouette, z, extent = crop_depth(frame.depth_m, mask)
        valid = z > 0
        if not np.any(valid):
            return None
        center = np.median(z[valid])
        observed_extent = extent * center / frame.K[[0, 1], [0, 1]]
        size_error = np.abs(self.extent - observed_extent).sum(axis=1) / diameter
        common = self.mask & valid
        shape_error = 1 - (self.mask & silhouette).sum(axis=(1, 2)) / (self.mask | silhouette).sum(axis=(1, 2))
        error = np.minimum(np.abs(self.depth - diameter - (z - center)), diameter)
        count = common.sum(axis=(1, 2))
        depth_error = (error * common).sum(axis=(1, 2)) / np.maximum(1, count)
        scores = shape_error + size_error + 4 * depth_error / diameter
        scores[count == 0] = np.inf
        indices = np.argsort(scores)[:16]
        return float(scores[indices[0]]), indices
