import numpy as np
import pytest

from mesh_pose.registration_inputs import make_crop_data_batch


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("rgb_max", [1, 255])
def test_network_crops_match_pinned_adapter(normalize, rgb_max):
    from spatialhub.models.foundationpose.adapter import make_crop_data_batch as reference

    rng = np.random.default_rng(13)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[:, :3, 3] = [[0, 0, 0.5], [0, 0.02, 0.6], [0.01, 0, 0.7]]
    xyz = rng.random((24, 32, 3), dtype=np.float32)
    xyz[:12, :, 2] = 0
    rendered_xyz = xyz[:10, :12][None].repeat(3, axis=0)

    class Renderer:
        def render(self, **kwargs):
            return rendered_xyz * (rgb_max / 255.0), None, None, rendered_xyz

    inputs = dict(render_size=(12, 10), ob_in_cams=poses, rgb=(xyz * rgb_max).astype(np.uint8),
                  depth=xyz[:, :, 2], intrinsics=np.float32([[30, 0, 16], [0, 30, 12], [0, 0, 1]]),
                  crop_ratio=1.2, xyz_map=xyz, model_diameter=0.25, normalize_xyz=normalize,
                  renderer=Renderer())
    for packed, original in zip(make_crop_data_batch(**inputs), reference(**inputs)):
        assert packed.flags.c_contiguous
        assert packed.dtype == original.dtype == np.float32
        np.testing.assert_array_equal(packed, original)
