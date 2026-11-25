"""Read the lit SpatialHub registration atlas directly into network input layout."""

from contextlib import ExitStack
from math import ceil, isqrt

import numpy as np

from . import _atlas


def render_inputs(renderer, poses, K, image_shape, boxes, size, diameter, normalize):
    from spatialhub.utils import intrinsics_to_projection_matrix, opencv_to_opengl

    height, width = image_shape
    count = len(poses)
    columns = isqrt(count - 1) + 1
    rows = ceil(count / columns)
    program = renderer.prog
    uniforms = dict(C=columns, R=rows, has_tex=int(renderer.has_tex), use_light=1,
                    w_ambient=0.8, w_diffuse=0.5, use_dir_light=1,
                    light_dir=(0., 0., 1.), light_pos=(0., 0., 0.), light_color=(1., 1., 1.))
    for name, value in uniforms.items():
        program[name].value = value
    if renderer.has_tex:
        renderer.tex_obj.use(0)
        program["tex_sampler"].value = 0

    projection = intrinsics_to_projection_matrix(K, height=height, width=width, znear=0.001, zfar=100.)
    left, right = boxes[:, 0], boxes[:, 2]
    top, bottom = height - boxes[:, 1], height - boxes[:, 3]
    crop = np.zeros((count, 4, 4), dtype=np.float32)
    crop[:, 0, 0] = width / (right - left)
    crop[:, 1, 1] = height / (top - bottom)
    crop[:, 0, 3] = (width - right - left) / (right - left)
    crop[:, 1, 3] = (height - top - bottom) / (top - bottom)
    crop[:, 2, 2] = crop[:, 3, 3] = 1.
    clip = crop @ (projection @ opencv_to_opengl(poses))
    tile_w, tile_h = size
    engine = renderer.engine
    with ExitStack() as resources:
        bindings = [(renderer.pos_buffer, "3f", "in_position"), (renderer.norm_buffer, "3f", "in_normal"),
                    (renderer.color_buffer, "3f", "in_color"), (renderer.uv_buffer, "2f", "in_uv")]
        for name, matrices in (("in_pose", poses), ("in_clip_mat", clip)):
            buffer = renderer.glctx.buffer(np.ascontiguousarray(matrices.transpose(0, 2, 1)), dynamic=True)
            resources.callback(buffer.release)
            bindings.append((buffer, "16f/i", name))
        vao = renderer.glctx.vertex_array(program, bindings, renderer.idx_buffer)
        resources.callback(vao.release)
        engine.render(vao, count, (rows, columns), (tile_h, tile_w), renderer.gbuffer_layout)
        # The pinned renderer stores RGBA color and camera-space XYZ in attachments 0 and 1.
        color, coords = engine._color_attachments[:2]
        packed = _atlas.pack(color.read(), coords.read(), np.ascontiguousarray(poses, np.float32),
                             count, columns, tile_h, tile_w, normalize, diameter / 2)
    return np.frombuffer(packed, np.float32).reshape(count, 6, tile_h, tile_w)
