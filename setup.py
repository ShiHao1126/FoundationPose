"""Build the registration atlas packer as part of the editable install."""

import sys
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "mesh_pose._atlas", ["src/mesh_pose/atlas.cpp"], language="c++",
    extra_compile_args=["/O2", "/fp:precise", "/openmp"] if sys.platform == "win32" else ["-O3"],
)])
