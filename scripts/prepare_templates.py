"""Prepare a mesh's detection and registration templates before opening a camera."""
from pathlib import Path

import numpy as np
from tap import Tap

from mesh_pose.backend import SpatialHubBackend
from mesh_pose.contracts import Config

ROOT = Path(__file__).resolve().parents[1]


class Args(Tap):
    mesh: Path = ROOT / "data/meshes/cover/cover.stl"
    mesh_unit: str = "mm"
    config: Path = ROOT / "configs/tracker.json"
    models: Path = ROOT / "configs/models.json"
    cache: Path = ROOT / "cache"


def main(args: Args):
    config = Config.load(args.config)
    # The backend requires pinhole intrinsics. Both template renderers use
    # their own canonical cameras, so the cache is independent of the live K.
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    backend = SpatialHubBackend(args.mesh, args.mesh_unit, K, config.backend,
                                config.geometry, args.models, args.cache)
    backend.close()
    print(f"Templates ready in {args.cache / 'templates'}")


if __name__ == "__main__":
    main(Args().parse_args())
