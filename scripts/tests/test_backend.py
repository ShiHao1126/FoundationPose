"""Detector crop coordinate contract without loading model weights."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from mesh_pose.backend import SpatialHubBackend
from mesh_pose.contracts import Config, RGBDFrame


def test_region_detection_uses_local_intrinsics_and_returns_full_image_masks(monkeypatch):
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    b = SpatialHubBackend.__new__(SpatialHubBackend)
    b.config, b.geometry, b.diameter_m = config.backend, config.geometry, .1
    cropped_mask = np.ones((20, 30), bool)
    proposals = SimpleNamespace(masks=cropped_mask[None], boxes=np.array([[0, 0, 30, 20]]), scores=np.array([.8]))
    def masks(rgb, **kwargs):
        assert rgb.shape == (20, 30, 3)
        if "num_max_dets" in kwargs:
            assert kwargs["num_max_dets"] == 1  # Local candidates are not cut to global top-K.
        return proposals
    b.segmentor = SimpleNamespace(generate_masks=masks)
    b.detector = SimpleNamespace(generate_masks=masks)
    monkeypatch.setitem(sys.modules, "spatialhub.structures", SimpleNamespace(InstanceSegmentation=SimpleNamespace))
    def supported(frame, mask, diameter, geometry):
        frame.validate()
        np.testing.assert_allclose(frame.K, [[100, 0, 10], [0, 100, 10], [0, 0, 1]])
        return True
    monkeypatch.setattr("mesh_pose.backend.candidate_has_depth_support", supported)
    frame = RGBDFrame(1, 1., np.zeros((40, 60, 3), np.uint8), np.ones((40, 60), np.float32),
                      np.array([[100, 0, 30], [0, 100, 20], [0, 0, 1]], np.float32))
    candidates = b.detect(frame, region=(20, 10, 50, 30))
    assert len(candidates) == 1 and candidates[0].mask.shape == (40, 60)
    assert candidates[0].mask[10:30, 20:50].all() and candidates[0].mask.sum() == 600
    np.testing.assert_allclose(frame.K[:2, 2], [30, 20])  # Input calibration stays untouched.


def test_global_detector_keeps_masks_beyond_registration_budget(monkeypatch):
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    b = SpatialHubBackend.__new__(SpatialHubBackend)
    b.config, b.geometry, b.diameter_m = config.backend, config.geometry, .1
    masks = np.zeros((7, 40, 60), bool)
    masks[:, 10:30, 20:40] = True
    proposals = SimpleNamespace(masks=masks, boxes=np.tile([20, 10, 40, 30], (7, 1)),
                                scores=np.linspace(.9, .4, 7))
    b.segmentor = SimpleNamespace(generate_masks=lambda *a, **k: proposals)
    def match(*a, **kwargs):
        assert kwargs["num_max_dets"] == 7
        return proposals
    b.detector = SimpleNamespace(generate_masks=match)
    monkeypatch.setitem(sys.modules, "spatialhub.structures", SimpleNamespace(InstanceSegmentation=SimpleNamespace))
    monkeypatch.setattr("mesh_pose.backend.candidate_has_depth_support", lambda *a: True)
    frame = RGBDFrame(1, 1., np.zeros((40, 60, 3), np.uint8), np.ones((40, 60), np.float32),
                      np.array([[100, 0, 30], [0, 100, 20], [0, 0, 1]], np.float32))
    assert config.backend.max_candidates == 3 and len(b.detect(frame)) == 7


def test_geometry_rescue_proposes_only_the_overall_winner_outside_the_budget(monkeypatch):
    from mesh_pose.contracts import Candidate

    b = SpatialHubBackend.__new__(SpatialHubBackend)
    b.config = SimpleNamespace(max_candidates=3, register_iterations=5)
    b.estimator = SimpleNamespace(rot_grid=np.eye(4)[None])
    b.diameter_m, b.geometry = .1, None
    masks = [np.eye(5, dtype=bool)[i:i+1] for i in range(5)]
    costs = [4., 3., 2., 1., .5]
    b.views = SimpleNamespace(match=lambda frame, mask, diameter: (costs[mask.argmax()], [0]))
    def register(*args):
        yield from ()
        return np.eye(4)
    monkeypatch.setattr("mesh_pose.registration.register_steps", register)
    monkeypatch.setattr("mesh_pose.backend.refine_depth_pose", lambda frame, pose, *args: pose)
    candidates = [Candidate(m, .8) for m in masks]
    assert [i for i, _ in b.guided_poses(None, candidates, True)] == [4]
    costs[1] = .1
    assert list(b.guided_poses(None, candidates, True)) == []
