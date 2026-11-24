"""SpatialHub resource owner and inference adapter; no camera or tracking state."""

from contextlib import ExitStack
from pathlib import Path
from hashlib import sha256
import json
import logging

import numpy as np

from .contracts import BackendConfig, Candidate, GeometryConfig, RGBDFrame
from .templates import ViewBank, save_cache
from .depth_renderer import DepthRenderer
from .geometry import candidate_has_depth_support, refine_depth_pose, touches_image_boundary

logger = logging.getLogger(__name__)


def download_models(manifest: Path, cache_dir: Path) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    models = json.loads(manifest.read_text(encoding="utf-8"))
    return {name: hf_hub_download(**spec, cache_dir=str(cache_dir / "huggingface"))
            for name, spec in models.items()}


class SpatialHubBackend:
    def __init__(self, mesh_path: Path, mesh_unit: str, K: np.ndarray,
                 config: BackendConfig, geometry: GeometryConfig, model_manifest: Path, cache_dir: Path):
        # PyTorch provides the CUDA 12 / cuDNN 9 DLLs for the pinned Windows ORT build.
        import torch
        from spatialhub import CNOS, DINOv2, FastSAM, FoundationPose
        from spatialhub.models.cnos.renderer import TemplateRenderer
        from spatialhub.utils import load_mesh, scale_mesh

        if mesh_unit not in {"m", "cm", "mm"}:
            raise ValueError("Mesh unit must be explicitly m, cm or mm")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the foundationpose environment")
        self._resources = ExitStack()
        self.config = config
        self.geometry = geometry
        self.K = np.asarray(K, dtype=np.float32).copy()
        mesh = scale_mesh(load_mesh(mesh_path), model_unit=mesh_unit)
        if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0:
            raise ValueError("Mesh must have finite vertices and triangular faces")
        self.T_centered_object = np.eye(4, dtype=np.float32)
        self.T_centered_object[:3, 3] = -np.asarray(mesh.bounding_box.centroid, dtype=np.float32)
        self.object_bounds_m = np.asarray(mesh.bounds)
        models = download_models(model_manifest, cache_dir)
        # Avoid large cuDNN benchmark workspaces on the 8 GB laptop GPU.
        providers = [("CUDAExecutionProvider", {
            "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
            "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        try:
            self.estimator = FoundationPose(
                intrinsics=self.K, model=mesh, model_unit="m", providers=providers,
                refinenet_onnx_path=models["refiner"], scorenet_onnx_path=models["scorer"])
            self._resources.callback(self.estimator.close)
            self.depth_renderer = DepthRenderer(self.estimator.glctx, self.estimator.renderer.pos_buffer,
                                                 self.estimator.renderer.idx_buffer)
            self._resources.callback(self.depth_renderer.close)
            self.diameter_m = float(self.estimator.model_diameter)
            self.descriptor = DINOv2(model_path=models["descriptor"], model_variant="vits14", providers=providers)
            self._resources.callback(self.descriptor.close)
            self.segmentor = FastSAM(model_path=models["segmentor"], model_variant="s", providers=providers)
            self._resources.callback(self.segmentor.close)
            sessions = {"refiner": self.estimator.refiner.ort_session,
                        "scorer": self.estimator.scorer.ort_session,
                        "descriptor": self.descriptor.session, "segmentor": self.segmentor.session}
            self.runtime = {name: session.get_providers() for name, session in sessions.items()}
            for name, session in sessions.items():
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError(f"{name} fell back to CPU; check CUDA/cuDNN installation")
                session.disable_fallback()
            fingerprint = sha256(mesh_path.read_bytes() + mesh_unit.encode() + model_manifest.read_bytes()
                                 + str(config.template_views).encode() + self.estimator.rot_grid.tobytes()).hexdigest()
            path = cache_dir / "templates" / f"v1-{fingerprint}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    self.views = ViewBank(data["depth"], data["extent"])
                    self.detector = CNOS(data["features"], self.descriptor, self.segmentor,
                                         chunk_size=config.descriptor_batch_size)
                logger.info("Loaded CAD templates from %s", path)
            else:
                logger.info("Building %d CAD templates; diameter %.4f m", config.template_views, self.diameter_m)
                renderer = TemplateRenderer(model=mesh, model_unit="m", width=256, height=256,
                                            max_chunk_size=16, ctx=self.estimator.glctx)
                try:
                    self.detector = CNOS.from_cad_model(
                        model=mesh, model_unit="m", descriptor=self.descriptor, segmentor=self.segmentor,
                        renderer=renderer, num_viewpoints=config.template_views,
                        radius=max(self.diameter_m * 2, 0.05), chunk_size=config.descriptor_batch_size)
                finally:
                    renderer.release()
                self.views = ViewBank.build(self.depth_renderer, self.estimator.rot_grid, self.diameter_m)
                save_cache(path, features=self.detector.reference_features,
                           depth=self.views.depth, extent=self.views.extent)
        except Exception:
            self._resources.close()
            raise

    def detect(self, frame: RGBDFrame, *, region: tuple[int, int, int, int] | None = None) -> list[Candidate]:
        from spatialhub.structures import InstanceSegmentation

        full_shape = frame.depth_m.shape
        if region is not None:
            x0, y0, x1, y1 = region
            if not (0 <= x0 < x1 <= full_shape[1] and 0 <= y0 < y1 <= full_shape[0]):
                raise ValueError("Detection region must be inside the image")
            K = frame.K.copy()
            K[0, 2] -= x0
            K[1, 2] -= y0
            frame = RGBDFrame(frame.sequence, frame.timestamp_s, frame.rgb[y0:y1, x0:x1],
                              frame.depth_m[y0:y1, x0:x1], K)
        proposals = self.segmentor.generate_masks(frame.rgb, conf_threshold=0.05, iou_threshold=0.9)
        keep = [i for i, mask in enumerate(proposals.masks)
                if (region is not None or not touches_image_boundary(mask))
                and candidate_has_depth_support(frame, mask, self.diameter_m, self.geometry)]
        logger.info("Detection: %d masks, %d with usable CAD-scale depth", len(proposals.masks), len(keep))
        if not keep:
            return []
        supported = InstanceSegmentation(masks=proposals.masks[keep], boxes=proposals.boxes[keep],
                                          scores=proposals.scores[keep])
        detections = self.detector.generate_masks(
            # Detection does not own the registration budget. A lower-ranked
            # whole-object mask can verify a pose initialized from a visible part.
            frame.rgb, proposals=supported, num_max_dets=len(keep),
            conf_threshold=self.config.min_detection_similarity, nms_iou_threshold=0.6)
        candidates = []
        for mask, score in zip(detections.masks, detections.scores):
            if region is not None:
                full_mask = np.zeros(full_shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = mask
                mask = full_mask
            candidates.append(Candidate(mask.astype(bool), float(score)))
        return candidates

    def guided_poses(self, frame, candidates, geometry_rescue=False):
        from .registration import register_steps

        ranked = []
        limit = self.config.max_candidates
        pool = candidates if geometry_rescue else candidates[:limit]
        for index, candidate in enumerate(pool):
            match = self.views.match(frame, candidate.mask, self.diameter_m)
            if match is not None:
                score, orientations = match
                ranked.append((score, index, orientations))
        ranked.sort(key=lambda item: item[0])
        if geometry_rescue:
            # Only the overall geometry winner can expand the appearance budget.
            if not ranked or ranked[0][1] < limit:
                return
            ranked = ranked[:1]
        for _, index, orientations in ranked:
            mask = candidates[index].mask
            steps = register_steps(self.estimator, frame, mask, self.config.register_iterations,
                                   self.estimator.rot_grid[orientations])
            pose = yield from steps
            yield index, refine_depth_pose(frame, pose, self.render_depth, self.geometry, mask)

    def register(self, frame: RGBDFrame, mask: np.ndarray) -> np.ndarray:
        self.estimator.reset_tracking()
        result = self.estimator.register(rgb=frame.rgb, depth=frame.depth_m, mask=mask,
                                         depth_unit="m", iteration=self.config.register_iterations)
        return refine_depth_pose(frame, result.pose[0], self.render_depth, self.geometry, mask)

    def register_steps(self, frame: RGBDFrame, mask: np.ndarray):
        from .registration import register_steps

        pose = yield from register_steps(self.estimator, frame, mask, self.config.register_iterations)
        return refine_depth_pose(frame, pose, self.render_depth, self.geometry, mask)

    def relocalize_seed(self, frame: RGBDFrame, mask: np.ndarray, seed: np.ndarray,
                        max_rotation_deg: float) -> np.ndarray:
        from .registration import relocalize_seed

        pose = relocalize_seed(self.estimator, frame, mask, seed, max_rotation_deg,
                               self.config.register_iterations)
        return refine_depth_pose(frame, pose, self.render_depth, self.geometry, mask)

    def track(self, frame: RGBDFrame, previous_pose: np.ndarray) -> np.ndarray:
        depth = self.estimator.depth_filter.apply(frame.depth_m, radius=2)
        h, w = depth.shape
        # Broadcast per-axis coordinates instead of building full UV grids.
        # Keep the upstream multiply-then-divide order and invalid-depth rule.
        xyz = np.empty((h, w, 3), dtype=np.float32)
        xyz[..., 0] = (np.arange(w, dtype=np.float32) - frame.K[0, 2]) * depth / frame.K[0, 0]
        xyz[..., 1] = (np.arange(h, dtype=np.float32)[:, None] - frame.K[1, 2]) * depth / frame.K[1, 1]
        xyz[..., 2] = depth
        xyz[depth < 0.001] = 0
        pose = self.estimator.refiner.predict(frame.rgb, depth, xyz, previous_pose[None],
                                              self.diameter_m, iteration=self.config.track_iterations)[0]
        return refine_depth_pose(frame, pose, self.render_depth, self.geometry)

    def render_depth(self, frame: RGBDFrame, pose: np.ndarray) -> np.ndarray:
        return self.depth_renderer.render(frame.K, frame.depth_m.shape, pose)

    def to_object_frame(self, pose: np.ndarray) -> np.ndarray:
        return pose @ self.T_centered_object

    def reset(self) -> None:
        self.estimator.reset_tracking()

    def close(self) -> None:
        self._resources.close()

