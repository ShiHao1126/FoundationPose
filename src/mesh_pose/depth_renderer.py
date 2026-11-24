"""Persistent camera-Z target for geometric checks, independent of neural crops."""

from contextlib import ExitStack

import moderngl
import numpy as np


class DepthRenderer:
    """Borrows the backend's GL context/mesh; owns only its scalar render target."""

    def __init__(self, ctx, positions, indices):
        self.ctx = ctx
        self._resources = ExitStack()
        self._target = ExitStack()
        self._resources.callback(self._target.close)
        self._size = None
        try:
            self.program = ctx.program(vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 pose;
                uniform mat4 clip;
                out float camera_z;
                out vec4 clip_position;
                void main() {
                    camera_z = (pose * vec4(in_position, 1.0)).z;
                    clip_position = clip * vec4(in_position, 1.0);
                    gl_Position = clip_position;
                }
            """, fragment_shader="""
                #version 330
                in float camera_z;
                in vec4 clip_position;
                out float depth;
                void main() {
                    if (clip_position.w <= 0.0 || abs(clip_position.x) > clip_position.w
                        || abs(clip_position.y) > clip_position.w) discard;
                    depth = camera_z;
                }
            """)
            self._resources.callback(self.program.release)
            self.vao = ctx.vertex_array(self.program, [(positions, "3f", "in_position")], indices)
            self._resources.callback(self.vao.release)
        except Exception:
            self.close()
            raise

    def render(self, K: np.ndarray, shape: tuple[int, int], pose: np.ndarray) -> np.ndarray:
        from spatialhub.utils import intrinsics_to_projection_matrix, opencv_to_opengl

        height, width = shape
        if self._size != (width, height):
            self._target.close()
            self.texture = self.ctx.texture((width, height), 1, dtype="f4")
            self._target.callback(self.texture.release)
            depth_buffer = self.ctx.depth_renderbuffer((width, height))
            self._target.callback(depth_buffer.release)
            self.fbo = self.ctx.framebuffer([self.texture], depth_buffer)
            self._target.callback(self.fbo.release)
            self._size = (width, height)
        projection = intrinsics_to_projection_matrix(K, height, width, znear=0.001, zfar=100.0)
        clip = projection @ opencv_to_opengl(pose)
        self.program["pose"].write(np.ascontiguousarray(pose.T, dtype=np.float32).tobytes())
        self.program["clip"].write(np.ascontiguousarray(clip.T, dtype=np.float32).tobytes())
        self.fbo.use()
        self.fbo.clear()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render(moderngl.TRIANGLES)
        return np.frombuffer(self.texture.read(), np.float32).reshape(shape)[::-1]

    def close(self) -> None:
        self._resources.close()
