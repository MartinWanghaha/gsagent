"""End-to-end robust association orchestration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .clustering import cluster_observations
from .config import RobustAssociationConfig
from .exporter import export_association
from .graph import build_association_graph
from .observations import build_view_observations, project_view, resolve_image
from .refinement import build_gaussian_consensus, refine_view


class RobustAssociationEngine:
    def __init__(
        self,
        projector,
        *,
        scene_dir: Path,
        images: str,
        raw_mask_dir: Path,
        point_cloud: Path,
        output_name: str,
        config: RobustAssociationConfig,
        visualize: bool,
        force: bool,
    ) -> None:
        config.validate()
        self.projector = projector
        self.scene_dir = Path(scene_dir)
        self.images = images
        self.raw_mask_dir = Path(raw_mask_dir)
        self.point_cloud = Path(point_cloud)
        self.output_name = output_name
        self.config = config
        self.visualize = visualize
        self.force = force

    def run(self) -> tuple[Path, Path | None]:
        viewpoints = self.projector.viewpoint_camera
        gaussian_xyz = self.projector.gaussians_xyz.detach().float().cpu().numpy()
        center = np.median(gaussian_xyz, axis=0)
        scene_scale = float(
            np.percentile(np.linalg.norm(gaussian_xyz - center, axis=1), 90)
        )
        scene_scale = max(scene_scale, 1e-6)
        observations = []
        observations_by_view = []
        camera_centers = []

        for view_index, viewpoint in enumerate(
            tqdm(viewpoints, desc="Build robust observations")
        ):
            viewpoint = viewpoint.to(self.projector.device)
            view_observations, _ = build_view_observations(
                self.projector,
                viewpoint,
                view_index=view_index,
                node_offset=len(observations),
                scene_dir=self.scene_dir,
                images=self.images,
                raw_mask_dir=self.raw_mask_dir,
                gaussian_xyz=gaussian_xyz,
                config=self.config,
            )
            observations.extend(view_observations)
            observations_by_view.append(view_observations)
            camera_centers.append(
                viewpoint.camera_center.detach().float().cpu().numpy()
            )
            if (view_index + 1) % 16 == 0:
                torch.cuda.empty_cache()

        candidates, selected, view_pairs = build_association_graph(
            observations_by_view,
            np.asarray(camera_centers, dtype=np.float32),
            config=self.config,
            scene_scale=scene_scale,
        )
        tracks = cluster_observations(
            observations,
            candidates,
            selected,
            config=self.config,
        )
        gaussian_labels, gaussian_confidence = build_gaussian_consensus(
            gaussian_xyz.shape[0],
            observations,
            margin_threshold=self.config.gaussian_label_margin,
        )

        def refined_views():
            """Yield one full-resolution result at a time for bounded RAM use."""
            for view_index, viewpoint in enumerate(
                tqdm(viewpoints, desc="Refine and export associated masks")
            ):
                viewpoint = viewpoint.to(self.projector.device)
                projection = project_view(self.projector, viewpoint, view_index)
                mask_path = self.raw_mask_dir / f"{viewpoint.image_name}.png"
                raw_labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if raw_labels is None or raw_labels.ndim != 2:
                    raise RuntimeError(f"Could not decode raw mask: {mask_path}")
                image_path = resolve_image(
                    self.scene_dir,
                    self.images,
                    viewpoint.image_name,
                )
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Could not decode RGB image: {image_path}")
                if image_bgr.shape[:2] != raw_labels.shape:
                    image_bgr = cv2.resize(
                        image_bgr,
                        (raw_labels.shape[1], raw_labels.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                yield refine_view(
                    raw_labels.astype(np.int32, copy=False),
                    image_bgr,
                    projection,
                    observations_by_view[view_index],
                    gaussian_labels,
                    gaussian_confidence,
                    config=self.config,
                )
                if (view_index + 1) % 16 == 0:
                    torch.cuda.empty_cache()

        return export_association(
            scene_dir=self.scene_dir,
            output_name=self.output_name,
            raw_mask_dir=self.raw_mask_dir,
            point_cloud=self.point_cloud,
            refined_views=refined_views(),
            observations=observations,
            tracks=tracks,
            candidate_edges=candidates,
            selected_edges=selected,
            config=self.config,
            visualize=self.visualize,
            force=self.force,
            extra_diagnostics={
                "scene_scale": scene_scale,
                "gaussians": int(gaussian_xyz.shape[0]),
                "camera_neighbor_pairs": len(view_pairs),
                "labeled_gaussians": int((gaussian_labels > 0).sum()),
                "gaussian_label_coverage": float(np.mean(gaussian_labels > 0)),
            },
        )
