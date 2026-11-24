"""Cooperative SpatialHub registration with the original 252-pose, 4-way tournament.

Each yield returns GPU ownership to the tracker so a fresh-frame recovery can
preempt an old global search. No rotations, refinement iterations or gates are
removed. The winning crops are retained between tournament levels.
"""

import numpy as np


def relocalize_seed(estimator, frame, mask, seed, max_rotation_deg, iterations):
    """Recenter from a fresh automatic mask and search the existing local motion bound."""
    from spatialhub.models.foundationpose.helper import guess_translation
    from spatialhub.utils import unproject_depth_map

    depth = estimator.depth_filter.apply(frame.depth_m, radius=2)
    xyz = unproject_depth_map(depth, estimator.K)
    grid = estimator.rot_grid
    relative = seed[:3, :3].T @ grid[:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    hypotheses = np.concatenate([seed[None], grid[angles <= max_rotation_deg]])
    hypotheses[:, :3, 3] = guess_translation(depth=depth, mask=mask, K=estimator.K)
    refined = estimator.refiner.predict(frame.rgb, depth, xyz, hypotheses,
                                        estimator.model_diameter, iteration=1)
    ordered, _ = estimator.scorer.predict(frame.rgb, depth, xyz, refined, estimator.model_diameter)
    return estimator.refiner.predict(frame.rgb, depth, xyz, ordered[:1],
                                      estimator.model_diameter, iteration=max(1, iterations - 1))[0]


def register_steps(estimator, frame, mask, iterations, grid=None):
    from spatialhub.models.foundationpose.adapter import make_crop_data_batch
    from spatialhub.models.foundationpose.helper import (
        egocentric_delta_pose_to_pose, guess_translation, rotvec_to_rotmat_np)
    from spatialhub.utils import unproject_depth_map

    depth = estimator.depth_filter.apply(frame.depth_m, radius=2)
    xyz = unproject_depth_map(depth, estimator.K)
    hypotheses = (estimator.rot_grid if grid is None else grid).copy()
    hypotheses[:, :3, 3] = guess_translation(depth=depth, mask=mask, K=estimator.K)
    refiner = estimator.refiner
    # Keep the original atlas dimensions and floating-point crop preparation.
    # Rendering subsets can perturb rasterization enough to change a tournament
    # winner on a symmetric object, even with the same list of rotations.
    a, b = make_crop_data_batch(
        render_size=refiner.input_resize, ob_in_cams=hypotheses, rgb=frame.rgb,
        depth=depth, intrinsics=refiner.K, crop_ratio=refiner.cfg["crop_ratio"],
        xyz_map=xyz, model_diameter=estimator.model_diameter,
        normalize_xyz=refiner.cfg["normalize_xyz"], renderer=refiner.renderer)
    yield
    translations, rotations = [], []
    for start in range(0, len(hypotheses), 4):
        outputs = refiner.ort_session.run(None, {
            refiner.input_names[0]: a[start:start + 4], refiner.input_names[1]: b[start:start + 4]})
        values = {o.name: value for o, value in zip(refiner.ort_session.get_outputs(), outputs)}
        translations.append(values["trans"])
        rotations.append(values["rot"])
        if start % 16 == 12:
            yield
    translation = np.concatenate(translations)
    rotation = np.concatenate(rotations)
    if refiner.cfg.get("normalize_xyz", False):
        translation *= estimator.model_diameter / 2.0
    else:
        translation = np.tanh(translation) * refiner.trans_normalizer
    rotation = rotvec_to_rotmat_np(np.tanh(rotation) * refiner.rot_normalizer).transpose(0, 2, 1)
    poses = egocentric_delta_pose_to_pose(hypotheses, translation, rotation)
    del a, b
    yield
    scorer = estimator.scorer

    def winner(a, b):
        outputs = scorer.ort_session.run(None, {scorer.input_names[0]: a, scorer.input_names[1]: b})
        values = dict(zip(scorer.output_names, outputs))
        return int(np.argmax(values.get("score_logit", outputs[0]).reshape(-1)))

    a, b = make_crop_data_batch(
        render_size=scorer.input_resize, ob_in_cams=poses, rgb=frame.rgb,
        depth=depth, intrinsics=scorer.K, crop_ratio=scorer.cfg["crop_ratio"],
        xyz_map=xyz, model_diameter=estimator.model_diameter,
        normalize_xyz=scorer.cfg["normalize_xyz"], renderer=scorer.renderer)
    yield
    while len(poses) > 1:
        winners = []
        for index in range(0, len(poses), 4):
            winners.append(index + winner(a[index:index + 4], b[index:index + 4]))
            if len(winners) % 4 == 0:
                yield
        poses, a, b = poses[winners], a[winners], b[winners]
    refined = estimator.refiner.predict(frame.rgb, depth, xyz, poses,
                                        estimator.model_diameter, iteration=max(1, iterations - 1))
    return refined[0]
