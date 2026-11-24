"""Behavioral checks with a deterministic renderer/backend; no models or hardware."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mesh_pose.contracts import Candidate, Config, RGBDFrame
from mesh_pose.geometry import candidate_has_depth_support, refine_depth_pose, verify_depth
from mesh_pose.tracker import Tracker


@pytest.fixture
def config():
    config = Config.load(Path(__file__).resolve().parents[2] / "configs" / "tracker.json")
    return replace(config, tracking=replace(config.tracking, search_interval_frames=1))


def frame(index=0, depth=None):
    return RGBDFrame(index, index / 30, np.zeros((40, 60, 3), np.uint8),
                     np.full((40, 60), 0.5, np.float32) if depth is None else depth,
                     np.array([[60, 0, 30], [0, 60, 20], [0, 0, 1]], np.float32))


def pose(x=0.0, angle=0.0):
    value = np.eye(4, dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    value[:2, :2] = [[c, -s], [s, c]]
    value[:3, 3] = [x, 0, 0.5]
    return value


def rendered_object(shape):
    depth = np.zeros(shape, np.float32)
    depth[1:-1, 1:-1] = .5
    return depth


class Backend:
    diameter_m = 0.1

    def __init__(self):
        self.next_pose = pose()
        self.detect_calls = 0
        self.register_calls = 0
        self.last_seed = None
        self.candidates = [Candidate(np.ones((40, 60), bool), 0.8)]
        self.registered_poses = []
        self.closed = False

    def detect(self, frame, *, region=None):
        self.detect_calls += 1
        self.last_region = region
        return self.candidates

    def guided_poses(self, frame, candidates):
        return iter(())

    def register(self, frame, mask):
        self.register_calls += 1
        return self.registered_poses.pop(0) if self.registered_poses else self.next_pose.copy()

    def register_steps(self, frame, mask):
        yield from ()
        return self.register(frame, mask)

    def track(self, frame, previous_pose):
        self.last_seed = previous_pose.copy()
        return self.next_pose.copy()

    def relocalize_seed(self, frame, mask, seed, max_rotation_deg):
        return self.next_pose.copy()

    def render_depth(self, frame, pose):
        return rendered_object(frame.depth_m.shape)

    def to_object_frame(self, pose):
        return pose.copy()

    def reset(self):
        self.last_seed = None

    def close(self):
        self.closed = True


def test_automatic_acquisition_requires_temporal_confirmation(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    first = tracker.step(frame(0))
    assert first.state == "CONFIRMING" and first.T_camera_object is None
    second = tracker.step(frame(1))
    assert second.state == "TRACKING" and second.T_camera_object is not None
    assert backend.detect_calls == 2 and backend.register_calls == 1


def test_loss_does_not_publish_stale_pose_and_recovery_is_global(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    tracker.step(frame(1))
    absent = tracker.step(frame(2, np.full((40, 60), 0.8, np.float32)))
    assert absent.state == "LOST" and absent.T_camera_object is None
    # Relocated and rotated beyond the local motion gate: full registration must still recover.
    backend.next_pose = pose(0.25, np.pi / 2)
    pending = tracker.step(frame(3))
    recovered = tracker.step(frame(4))
    assert pending.T_camera_object is None and recovered.state == "TRACKING"
    np.testing.assert_allclose(recovered.T_camera_object, backend.next_pose)
    assert backend.register_calls == 2


def test_wrong_depth_candidate_is_rejected(config):
    tracker = Tracker(Backend(), config)
    result = tracker.step(frame(0, np.full((40, 60), 0.8, np.float32)))
    assert result.reason == "no_verified_candidate" and result.T_camera_object is None


def test_ambiguity_does_not_arbitrarily_select_an_instance(config):
    backend = Backend()
    backend.candidates *= 2
    backend.registered_poses = [pose(-0.1), pose(0.1)]
    result = Tracker(backend, config).step(frame())
    assert result.reason == "ambiguous_instances" and result.T_camera_object is None


def test_weaker_distractor_does_not_veto_a_clearly_better_candidate(config):
    backend = Backend()
    backend.candidates *= 2
    backend.registered_poses = [pose(-0.1), pose(0.1)]
    def render(frame, pose):
        depth = rendered_object(frame.depth_m.shape)
        if pose[0, 3] > 0:
            depth[1:12, 1:-1] = 0.512
        return depth
    backend.render_depth = render
    result = Tracker(backend, config).step(frame())
    assert result.state == "CONFIRMING" and result.reason != "ambiguous_instances"


def test_partial_fit_cannot_initialize_when_it_is_the_only_candidate(config):
    backend = Backend()
    def render(frame, pose):
        depth = rendered_object(frame.depth_m.shape)
        depth[1:12, 1:-1] = 0.512
        return depth
    backend.render_depth = render
    result = Tracker(backend, config).step(frame())
    assert result.T_camera_object is None and result.quality.reason == "weak_registration_geometry"


def test_tracking_starts_from_selected_candidate_not_last_backend_candidate(config):
    backend = Backend()
    backend.candidates *= 2
    first, second = pose(), pose(0.01)
    backend.registered_poses = [first, second]
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    np.testing.assert_array_equal(backend.last_seed, first)


def test_zero_depth_and_invalid_timestamps_clear_tracking(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    lost = tracker.step(frame(2, np.zeros((40, 60), np.float32)))
    assert lost.T_camera_object is None and lost.reason == "missing_depth"
    with pytest.raises(ValueError, match="increasing"):
        tracker.step(frame(1))
    assert tracker.state == "LOST"


def test_frame_gap_requires_strong_local_reconfirmation(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    result = tracker.step(frame(40))
    assert result.state == "CONFIRMING" and backend.register_calls == 1
    assert result.reason == "confirming_local_recovery" and result.T_camera_object is None
    assert tracker.step(frame(41)).state == "TRACKING"


def test_slow_registration_requires_fresh_mask_confirmation(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    first = tracker.step(frame())
    assert first.T_camera_object is None
    # Fifteen seconds passed while registering an old frame. Do not register it
    # over and over: use the unpublished seed, but re-detect on the fresh frame.
    result = tracker.step(frame(450))
    assert result.state == "TRACKING" and result.T_camera_object is not None
    assert backend.register_calls == 1 and backend.detect_calls == 2
    assert backend.last_region == (0, 0, 60, 40)


@pytest.mark.parametrize("failure", ["absent_mask", "weak_fit", "moved_away"])
def test_slow_registration_cannot_publish_an_unconfirmed_stale_pose(config, failure):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    if failure == "absent_mask":
        backend.candidates = []
    elif failure == "weak_fit":
        def render(frame, pose):
            depth = rendered_object(frame.depth_m.shape)
            depth[1:12, 1:-1] = .512  # Passes tracking gates, fails registration gates.
            return depth
        backend.render_depth = render
    else:
        backend.next_pose = pose(.25)
    result = tracker.step(frame(450))
    assert result.state == "LOST" and result.T_camera_object is None


def test_periodic_detection_catches_geometrically_plausible_drift(config):
    config = replace(config, tracking=replace(config.tracking, detection_check_interval_frames=2))
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    backend.candidates = []
    pending = tracker.step(frame(2))
    assert pending.state == "CONFIRMING" and pending.T_camera_object is None
    result = tracker.step(frame(3))
    assert result.reason == "detection_check_failed" and result.T_camera_object is None


def test_single_detector_miss_withholds_pose_then_recovers_without_global_registration(config):
    config = replace(config, tracking=replace(config.tracking, detection_check_interval_frames=2))
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    masks = backend.candidates
    backend.candidates = []
    pending = tracker.step(frame(2))
    assert pending.reason == "confirming_detection" and pending.T_camera_object is None
    backend.candidates = masks
    recovered = tracker.step(frame(3))
    assert recovered.state == "TRACKING" and recovered.T_camera_object is not None
    assert backend.register_calls == 1 and backend.detect_calls == 4


def test_occlusion_during_detection_recheck_loses_immediately(config):
    config = replace(config, tracking=replace(config.tracking, detection_check_interval_frames=2))
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    tracker.step(frame(1))
    backend.candidates = []
    tracker.step(frame(2))
    lost = tracker.step(frame(3, np.full((40, 60), .3, np.float32)))
    assert lost.state == "LOST" and lost.T_camera_object is None


def test_latest_local_recovery_preempts_old_global_search(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    tracker.step(frame(1))
    tracker.step(frame(2, np.full((40, 60), .8, np.float32)))
    closed = []
    def staged(image, mask):
        try:
            for _ in range(10):
                yield
            return pose()
        finally:
            closed.append(True)
    backend.register_steps = staged
    searching = tracker.step(frame(3, np.full((40, 60), .8, np.float32)), cooperative=True)
    assert searching.reason == "global_search_in_progress" and searching.T_camera_object is None
    pending = tracker.step(frame(4))
    assert pending.reason == "confirming_local_recovery" and pending.T_camera_object is None
    assert closed == [True]
    recovered = tracker.step(frame(5))
    assert recovered.state == "TRACKING" and backend.register_calls == 1


def test_local_recovery_requires_strong_geometry_and_fresh_mask(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    tracker.step(frame(1))
    tracker.step(frame(2, np.full((40, 60), .3, np.float32)))
    pending = tracker.step(frame(3))
    assert pending.reason == "confirming_local_recovery" and pending.T_camera_object is None
    backend.candidates = []
    rejected = tracker.step(frame(4))
    assert rejected.reason == "fresh_confirmation_failed" and rejected.T_camera_object is None
    backend.candidates = [Candidate(np.ones((40, 60), bool), .8)]
    def weak_render(image, pose):
        depth = rendered_object(image.depth_m.shape)
        depth[1:12, 1:-1] = .512
        return depth
    backend.render_depth = weak_render
    rejected = tracker.step(frame(5))
    assert rejected.T_camera_object is None and rejected.reason == "no_verified_candidate"


def test_staged_global_search_finishes_after_gaps_but_cannot_publish_old_frame(config):
    backend = Backend()
    def staged(image, mask):
        yield
        yield
        return pose()
    backend.register_steps = staged
    tracker = Tracker(backend, config)
    assert tracker.step(frame(0), cooperative=True).reason == "global_search_in_progress"
    assert tracker.step(frame(60), cooperative=True).reason == "global_search_in_progress"
    assert tracker.step(frame(120), cooperative=True).state == "CONFIRMING"
    backend.candidates = []
    result = tracker.step(frame(121))
    assert result.T_camera_object is None and result.reason == "fresh_confirmation_failed"
    tracker.close()


def test_clipped_mesh_cannot_register_even_with_matching_visible_depth_and_mask(config):
    backend = Backend()
    def clipped(image, pose):
        depth = rendered_object(image.depth_m.shape)
        depth[-1, 10:30] = .5
        return depth
    backend.render_depth = clipped
    result = Tracker(backend, config).step(frame())
    assert result.reason == "no_verified_candidate"
    assert result.quality.reason == "truncated_registration" and result.T_camera_object is None


def test_old_initialization_is_relocated_on_new_mask_then_confirmed_on_another_frame(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    real_track = backend.track
    backend.track = lambda image, seed: np.full((4, 4), np.nan)
    backend.next_pose = pose(.02, .5)
    refreshed = tracker.step(frame(450))
    assert refreshed.reason == "confirming_refreshed_initialization"
    assert refreshed.T_camera_object is None and backend.register_calls == 1
    backend.track = real_track
    confirmed = tracker.step(frame(451))
    assert confirmed.state == "TRACKING"
    np.testing.assert_allclose(confirmed.T_camera_object, backend.next_pose)


def test_refreshed_initialization_cannot_bypass_motion_or_strong_geometry(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    backend.track = lambda image, seed: np.full((4, 4), np.nan)
    backend.next_pose = pose(.2)
    rejected = tracker.step(frame(450))
    assert rejected.state == "LOST" and rejected.T_camera_object is None


def test_strong_confirmation_can_use_full_image_mask_when_crop_misses(config):
    backend = Backend()
    mask = np.zeros((40, 60), bool)
    mask[12:28, 22:38] = True
    backend.candidates = [Candidate(mask, .8)]
    backend.render_depth = lambda image, pose: mask.astype(np.float32) * .5
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    seen_regions = []
    def detect(image, *, region=None):
        seen_regions.append(region)
        return [] if region is not None else backend.candidates
    backend.detect = detect
    confirmed = tracker.step(frame(1))
    assert confirmed.state == "TRACKING"
    assert seen_regions[0] is not None and seen_regions[1] is None


def test_lower_ranked_whole_mask_verifies_part_seed_without_more_registrations(config):
    backend = Backend()
    part = np.zeros((40, 60), bool)
    part[1:10, 1:-1] = True
    whole = rendered_object(part.shape) > 0
    backend.candidates = [Candidate(part.copy(), .9 - i * .05) for i in range(6)]
    backend.candidates.append(Candidate(whole, .4))
    tracker = Tracker(backend, config)
    pending = tracker.step(frame(0))
    assert pending.state == "CONFIRMING" and pending.quality.mask_iou == 1.
    assert backend.register_calls == config.backend.max_candidates
    assert tracker.step(frame(1)).state == "TRACKING"


def test_backend_fault_is_fatal_and_clears_state(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame())
    def fail(*args):
        raise RuntimeError("GPU failed")
    backend.track = fail
    with pytest.raises(RuntimeError, match="GPU failed"):
        tracker.step(frame(1))
    assert tracker.state == "LOST"
    tracker.close()
    tracker.close()
    assert backend.closed


def test_depth_verifier_accepts_partial_occlusion_but_not_complete_occlusion(config):
    rendered = np.full((40, 60), 0.5, np.float32)
    observed = rendered.copy()
    observed[:15] = 0.3
    assert verify_depth(observed, rendered, config.geometry).accepted
    observed[:] = 0.3
    quality = verify_depth(observed, rendered, config.geometry)
    assert not quality.accepted and quality.reason == "insufficient_visibility"


def test_depth_holes_cannot_count_as_inliers(config):
    rendered = np.full((40, 60), 0.5, np.float32)
    observed = rendered.copy()
    observed[:30] = 0
    assert not verify_depth(observed, rendered, config.geometry).accepted


def test_mask_check_rejects_wrong_silhouette(config):
    depth = np.full((40, 60), 0.5, np.float32)
    mask = np.zeros((40, 60), bool)
    mask[:5] = True
    assert verify_depth(depth, depth, config.geometry, mask).reason == "mask_mismatch"


def test_candidate_budget_excludes_missing_depth_and_wrong_physical_size(config):
    image = frame()
    target = np.zeros(image.depth_m.shape, bool)
    target[10:24, 20:34] = True
    assert candidate_has_depth_support(image, target, 0.1, config.geometry)
    no_depth = replace(image, depth_m=np.zeros_like(image.depth_m))
    assert not candidate_has_depth_support(no_depth, target, 0.1, config.geometry)
    room = np.ones(image.depth_m.shape, bool)
    assert not candidate_has_depth_support(image, room, 0.1, config.geometry)
    assert not candidate_has_depth_support(image, target, 1.0, config.geometry)


def test_local_depth_refinement_corrects_offset_without_inventing_planar_motion(config):
    image = frame()
    initial = pose(x=0.1)
    initial[2, 3] = 0.508
    def render_plane(frame, pose):
        return np.full(frame.depth_m.shape, pose[2, 3], np.float32)
    refined = refine_depth_pose(image, initial, render_plane, config.geometry)
    assert abs(refined[2, 3] - 0.5) < 1e-5
    np.testing.assert_allclose(refined[:2, 3], initial[:2, 3], atol=1e-6)
    np.testing.assert_allclose(refined[:3, :3], initial[:3, :3], atol=1e-5)
    missing = replace(image, depth_m=np.full_like(image.depth_m, 2.0))
    np.testing.assert_array_equal(refine_depth_pose(missing, initial, render_plane, config.geometry), initial)


@pytest.mark.parametrize("bad", [np.full((4, 4), np.nan), np.diag([-1.0, 1.0, 1.0, 1.0])])
def test_invalid_pose_is_never_exported(config, bad):
    backend = Backend()
    backend.next_pose = bad
    result = Tracker(backend, config).step(frame())
    assert result.T_camera_object is None and result.quality.reason == "invalid_pose"



def test_guided_region_requires_full_grid_and_fresh_confirmation(config):
    backend = Backend()
    backend.guided_poses = lambda *_: iter([None, (0, pose())])
    tracker = Tracker(backend, config)
    seed = tracker.step(frame(0))
    assert seed.state == "CONFIRMING" and seed.T_camera_object is None
    assert tracker.step(frame(1)).state == "TRACKING"
    assert backend.register_calls == 1


def test_distinct_guided_instances_remain_ambiguous(config):
    backend = Backend()
    backend.candidates *= 2
    backend.registered_poses = [pose(), pose(0.07)]
    backend.guided_poses = lambda *_: iter([(0, pose()), (1, pose(0.07))])
    result = Tracker(backend, config).step(frame())
    assert result.reason == "ambiguous_instances" and result.T_camera_object is None


def test_rejected_guided_pose_falls_back_to_full_grid(config):
    backend = Backend()
    backend.guided_poses = lambda *_: iter([(0, np.zeros((4, 4)))])
    result = Tracker(backend, config).step(frame())
    assert result.state == "CONFIRMING" and backend.register_calls == 1


def test_live_shortlist_miss_runs_full_search_on_new_input(config):
    backend = Backend()
    backend.guided_poses = lambda *_: iter([(0, np.zeros((4, 4)))])
    tracker = Tracker(backend, config)
    assert tracker.step(frame(0), cooperative=True).reason == "global_search_in_progress"
    assert tracker.step(frame(1), cooperative=True).reason == "retry_full_on_fresh_frame"
    assert backend.register_calls == 0
    assert tracker.step(frame(2), cooperative=True).state == "CONFIRMING"
    assert backend.register_calls == 1


def test_mask_recovery_uses_prior_and_requires_another_frame(config):
    backend = Backend()
    tracker = Tracker(backend, config)
    tracker.step(frame(0))
    tracker.step(frame(1))
    tracker.step(frame(2, np.full((40, 60), 0.3, np.float32)))
    # Local tracking misses. The automatic mask gives a corrected center.
    backend.track = lambda *_: np.zeros((4, 4))
    backend.relocalize_seed = lambda *_: pose(0.01)
    pending = tracker.step(frame(3))
    assert pending.reason == "confirming_mask_recovery" and pending.T_camera_object is None
    assert backend.register_calls == 1
    backend.track = lambda *_: pose(0.01)
    confirmed = tracker.step(frame(4))
    assert confirmed.state == "TRACKING" and confirmed.T_camera_object is not None
