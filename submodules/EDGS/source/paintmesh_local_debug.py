"""Fixed-view local training montages; no global PGSR debug settings changed."""
from pathlib import Path

import numpy as np
import torch

from source.pgsr_debug import _opencv, _rgb_panel
from source.paintmesh_local_losses import unit_normal


class LocalGeometryDebug:
    def __init__(self, root, config, target, alpha_min):
        self.config = config
        self.root = Path(root) / "debug"
        self.alpha_min = alpha_min
        depth = target["depth"].detach().cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0)
        self.depth_range = (float(depth[valid].min()), float(depth[valid].max())) if valid.any() else (0., 1.)

    def due(self, completed_steps):
        return (self.config["enabled"] and completed_steps >= self.config["from_step"] and
                (completed_steps - self.config["from_step"]) % self.config["interval"] == 0)

    @torch.no_grad()
    def save(self, package, target, metrics, completed_steps, *, final=False):
        if not self.config["enabled"]:
            return
        from local_geometry_io import atomic_write, write_json
        cv2 = _opencv()
        root = self.root.resolve()
        if not root.is_relative_to(self.root.parent.resolve()):
            raise ValueError("local debug output escapes local_geometry directory")
        rgb = package["render"]
        height, width = rgb.shape[-2:]
        alpha = package["rendered_alpha"].squeeze(0).detach().cpu().numpy()
        depth = package["plane_depth"].squeeze(0).detach().cpu().numpy()
        pred, pred_valid = unit_normal(package["rendered_normal"])
        normal = pred.permute(1, 2, 0).cpu().numpy()
        reference = target["normal"].detach().permute(1, 2, 0).cpu().numpy()
        target_valid = target["normal_valid"].detach().cpu().numpy().astype(bool)
        visible = pred_valid.cpu().numpy() & np.isfinite(depth) & (depth > 0) & (alpha >= self.alpha_min)
        mask = target["mask"].detach().cpu().numpy().astype(bool)

        def colors(array, valid, low, high):
            scaled = np.nan_to_num((array - low) / max(high - low, 1e-6))
            encoded = np.rint(np.clip(scaled, 0, 1) * 255).astype(np.uint8)
            result = cv2.applyColorMap(encoded, cv2.COLORMAP_JET)
            result[~valid] = 0
            return result

        def normals(array, valid):
            result = np.rint(np.clip(np.nan_to_num(array) * .5 + .5, 0, 1) * 255).astype(np.uint8)
            result[~valid] = 0
            return np.ascontiguousarray(result[..., ::-1])

        target_depth = target["depth"].detach().cpu().numpy()
        angular = np.degrees(np.arccos(np.clip((normal * reference).sum(-1), -1, 1)))
        alpha_panel = colors(alpha, np.isfinite(alpha), 0, 1)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(alpha_panel, contours, -1, (255, 255, 255), 1)
        panels = [
            ("Completed RGB", _rgb_panel(target["rgb"], "target")),
            ("Rendered RGB", _rgb_panel(rgb, "render")),
            ("Completed depth", colors(target_depth, np.isfinite(target_depth) & (target_depth > 0), *self.depth_range)),
            ("Rendered depth", colors(depth, visible, *self.depth_range)),
            ("LaMa normal", normals(reference, target_valid)),
            ("Rendered normal", normals(normal, visible)),
            ("Normal error 0-180 deg (hole)", colors(angular, visible & target_valid & mask, 0, 180)),
            ("Alpha 0-1 / hole boundary", alpha_panel),
        ]
        tiles = []
        for title, panel in panels:
            # Small synthetic images are enlarged only for readable captions.
            tile_width = max(width, 280)
            panel = cv2.resize(panel, (tile_width, max(height, 140)), interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(panel, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            cv2.putText(tile, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            tiles.append(tile)
        montage = np.concatenate((np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)), axis=0)
        montage = cv2.copyMakeBorder(montage, 45, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        values = {key: float(value) for key, value in metrics.items()}
        caption = (f"{'Final gated' if final else 'Optimization'} | updates={completed_steps} "
                   f"view={self.config['view_index']:05d} ramp={values['ramp']:.3f} loss={values['total']:.6f}")
        cv2.putText(montage, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(montage, f"depth range={self.depth_range} (fixed scene units); black=invalid", (8, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        stem = "final" if final else f"step_{completed_steps:06d}"
        destination = self.root / f"{stem}_view_{self.config['view_index']:05d}.jpg"
        ok, encoded = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, self.config["jpeg_quality"]])
        if not ok:
            raise OSError("could not encode local geometry debug montage")
        atomic_write(destination, lambda stream: stream.write(encoded.tobytes()))
        write_json(destination.with_suffix(".json"), dict(completed_steps=completed_steps,
            state="final_gated" if final else "post_update", view_index=self.config["view_index"],
            depth_range=self.depth_range, metrics=values))
