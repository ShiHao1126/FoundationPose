"""Prepare contiguous RGB-XYZ inputs for batched CAD registration."""

import cv2
import numpy as np

from .registration_atlas import render_inputs


def _pack(rgb, xyz, centers, diameter, normalize_xyz):
    """Pack NHWC crops directly into the float32 NCHW layout consumed by ONNX."""
    n, h, w, _ = rgb.shape
    tensor = np.empty((n, 6, h, w), dtype=np.float32)
    color, coords = tensor[:, :3], tensor[:, 3:]
    color[:] = rgb.transpose(0, 3, 1, 2)
    if rgb.max() > 1:
        color /= 255.0
    np.subtract(xyz.transpose(0, 3, 1, 2), centers, out=coords)
    if normalize_xyz:
        coords /= float(diameter / 2.0)
        invalid_depth = xyz[:, None, :, :, 2] < 0.001
        coords[invalid_depth | (np.abs(coords) >= 2)] = 0
    return tensor


def make_crop_data_batch(*, render_size, ob_in_cams, rgb, depth, intrinsics, crop_ratio,
                         xyz_map, model_diameter, normalize_xyz, renderer):
    from spatialhub.models.foundationpose.helper import compute_crop_window_tf_batch

    height, width = depth.shape
    transforms, boxes = compute_crop_window_tf_batch(
        H=height, W=width, poses=ob_in_cams, intrinsics=intrinsics, crop_ratio=crop_ratio,
        out_size=render_size, model_diameter=model_diameter)
    synthetic = render_inputs(renderer, ob_in_cams, intrinsics, depth.shape, boxes, render_size,
                              model_diameter, normalize_xyz)
    centers = np.asarray(ob_in_cams[:, :3, 3, None, None], dtype=np.float32)
    shared = np.all(centers == centers[:1]) and np.all(transforms == transforms[:1])
    if shared:
        transforms = transforms[:1]
        centers = centers[:1]

    def warp(image, interpolation):
        crops = np.empty((len(transforms), render_size[1], render_size[0], 3), dtype=np.float32)
        for index, transform in enumerate(transforms):
            crops[index] = cv2.warpPerspective(image, transform, render_size, flags=interpolation)
        return crops

    observed = _pack(warp(rgb, cv2.INTER_LINEAR), warp(xyz_map, cv2.INTER_NEAREST),
                     centers, model_diameter, normalize_xyz)
    if shared:
        observed = np.broadcast_to(observed, synthetic.shape)
    return synthetic, observed
