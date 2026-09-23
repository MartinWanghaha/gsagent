"""Run-local PGSR previews that do not require disabled external targets."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from source.paintmesh_local_losses import unit_normal


@torch.no_grad()
def save_debug(root, package, target, metrics, step, final=False):
    from edgs_inpaint_io import atomic_write, write_json
    normal, valid = unit_normal(package["rendered_normal"])
    arrays = dict(rgb=package["render"].permute(1, 2, 0).cpu().numpy(),
                  depth=package["plane_depth"].squeeze(0).cpu().numpy(),
                  normal=normal.permute(1, 2, 0).cpu().numpy(),
                  alpha=package["rendered_alpha"].squeeze(0).cpu().numpy())
    root = Path(root) / "debug"
    stem = "final" if final else f"step_{step:06d}"
    atomic_write(root / (stem + ".npz"), lambda f: np.savez_compressed(f, **arrays))
    visible = valid.cpu().numpy() & (arrays["alpha"] > .01)
    def normal_image(value, mask):
        im = np.clip(value * .5 + .5, 0, 1)
        im[~mask] = 0
        return im
    def depth_image(value):
        valid = np.isfinite(value) & (value > 0)
        im = np.zeros((*value.shape, 3))
        if valid.any():
            lo, hi = np.percentile(value[valid], [2, 98])
            v = np.clip((value - lo) / max(hi - lo, 1e-6), 0, 1)
            im[valid] = np.stack((v, 1 - np.abs(2 * v - 1), 1 - v), -1)[valid]
        return im
    empty = np.zeros_like(arrays["rgb"])
    panels = [
        ("Completed RGB", target["rgb"].permute(1, 2, 0).cpu().numpy()),
        ("Rendered RGB", arrays["rgb"]),
        ("Completed depth" if "depth" in target else "Depth target: N/A", depth_image(target["depth"].cpu().numpy()) if "depth" in target else empty),
        ("Rendered depth (preview range)", depth_image(arrays["depth"])),
        ("LaMa normal" if "normal" in target else "Normal target: N/A", normal_image(target["normal"].permute(1, 2, 0).cpu().numpy(), target["normal_valid"].cpu().numpy()) if "normal" in target else empty),
        ("Rendered normal", normal_image(arrays["normal"], visible)),
        ("Hole mask", np.repeat(target["mask"].cpu().numpy()[..., None], 3, -1)),
        ("Rendered alpha", np.repeat(arrays["alpha"][..., None], 3, -1)),
    ]
    width, height = max(280, arrays["rgb"].shape[1]), max(140, arrays["rgb"].shape[0])
    canvas = Image.new("RGB", (width * 4, (height + 24) * 2 + 28), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"PGSR joint | updates={step} | single-view loss={float(metrics['total']):.6f}", fill="white")
    for index, (title, value) in enumerate(panels):
        x, y = (index % 4) * width, 28 + (index // 4) * (height + 24)
        tile = Image.fromarray(np.rint(np.clip(np.nan_to_num(value), 0, 1) * 255).astype(np.uint8)).resize((width, height))
        canvas.paste(tile, (x, y + 24))
        draw.text((x + 5, y + 5), title, fill="white")
    atomic_write(root / (stem + ".jpg"), lambda f: canvas.save(f, format="JPEG", quality=95))
    write_json(root / (stem + ".json"), dict(completed_steps=step,
        metrics={k: float(v) for k, v in metrics.items()}, targets=sorted(target)))
