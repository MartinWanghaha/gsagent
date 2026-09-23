"""Export the exact Stage 5a retained rows without running training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "scripts/paintmesh"))


def main():
    import numpy as np
    import torch
    from plyfile import PlyData
    from gaussian_renderer import GaussianModel
    from edit_object_inpaint import select_inpaint_masks, get_projected_gaussians
    from utils.virtual_camera_manifest import load_virtual_camera_manifest, virtual_views_from_manifest
    from PIL import Image
    from types import SimpleNamespace
    from local_geometry_io import atomic_write, read_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-ply", "classifier", "inpaint-config", "output", "camera", "mask", "support"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed-frame", type=int, required=True)
    args = parser.parse_args()
    config = read_json(args.inpaint_config)
    vertex = PlyData.read(args.source_ply)["vertex"].data
    degree = int(np.sqrt(sum(n.startswith("f_rest_") for n in vertex.dtype.names) / 3 + 1)) - 1
    model = GaussianModel(degree)
    model.load_ply(str(args.source_ply))
    state = torch.load(args.classifier, map_location="cuda", weights_only=True)
    classifier = torch.nn.Conv2d(model.num_objects, state["weight"].shape[0], 1).cuda()
    classifier.load_state_dict(state)
    masks = select_inpaint_masks(model, classifier, config["select_obj_id"], config["removal_thresh"])
    removed = torch.stack([v["mask3d"].bool().reshape(-1) for v in masks.values()]).any(0)
    rows = torch.where(~removed)[0].cpu().numpy()
    manifest = load_virtual_camera_manifest(args.camera)
    from local_geometry_io import identity
    if identity(manifest) != read_json(args.camera)["artifact_id"]:
        raise ValueError("camera manifest identity mismatch")
    first = manifest["cameras"][0]
    base = SimpleNamespace(world_view_transform=torch.eye(4, device="cuda"),
        image_width=first["image_width"], image_height=first["image_height"])
    view = virtual_views_from_manifest(base, manifest)[args.seed_frame]
    view.objects = torch.from_numpy(np.array(Image.open(args.mask).convert("L"))).cuda()
    gate = get_projected_gaussians(model, view, args.support)
    atomic_write(args.output, lambda stream: np.savez_compressed(stream,
        retained_rows=rows, xyz=model.get_xyz.detach().cpu().numpy()[rows],
        opacity=model.get_opacity.detach().cpu().numpy()[rows, 0],
        scale=model.get_scaling.detach().cpu().numpy()[rows],
        projection=gate["gate_projection"].cpu().numpy(), mask=gate["gate_mask"].astype(bool),
        distance_threshold=np.array(gate["gate_distance_threshold"])))


if __name__ == "__main__":
    main()
