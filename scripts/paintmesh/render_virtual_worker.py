"""Isolated backend adapters. Only the selected project's modules are imported."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from render_virtual_views import REPO, parser, render_roots
from virtual_render_io import FrameWriter, identity, read_json, sha256


def camera_module():
    # Reuse the exact manifest validation/reconstruction. Its only project
    # dependency is utils.graphics_utils, equivalent in both selected projects.
    path = REPO / "submodules/Inpaint360GS/utils/virtual_camera_manifest.py"
    spec = importlib.util.spec_from_file_location("paintmesh_virtual_cameras", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cameras(args):
    module = camera_module()
    normalized = module.load_virtual_camera_manifest(
        args.camera_manifest, expected_iteration=args.iteration
    )
    original = read_json(args.camera_manifest)
    if identity(normalized) != original["artifact_id"]:
        raise ValueError("camera manifest content does not match artifact_id")
    return module, original


def input_records(args, ply, variant, configuration, classifier=None):
    files = {"ply": ply, "configuration": configuration}
    if classifier is not None:
        files["classifier"] = classifier
    return {
        "camera_sha256": sha256(args.camera_manifest),
        "removal_variant": variant,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in files.items()
        },
    }


def model_files(args):
    # Matches the original virtual_pose.py: remove target + temporary occluders
    # for completion, not the published target-only mesh model.
    return (
        args.model_path
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply",
        args.model_path
        / "point_cloud_object_removal"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply",
    )


def native(args):
    import torch
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from render import render_set
    from tools.virtual_pose import render_set_removal_stage

    cfg_path = args.model_path / "cfg_args"
    expression = ast.parse(cfg_path.read_text(), mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "Namespace"
        or expression.args
    ):
        raise ValueError("cfg_args must be a literal Namespace")
    cfg = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in expression.keywords
        if kw.arg is not None
    }
    cfg.update(
        model_path=str(args.model_path),
        source_path=str(args.source_path),
        resolution=args.resolution,
        images=args.images,
        init_mode="sparse",
        train_distill=False,
    )
    dataset = SimpleNamespace(**cfg)
    pipeline = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    module, cameras = load_cameras(args)
    views = module.virtual_views_from_manifest(scene.getTrainCameras()[0], cameras)
    classifier_path = (
        args.model_path
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "classifier.pth"
    )
    state = torch.load(classifier_path, map_location="cuda", weights_only=True)
    classifier = torch.nn.Conv2d(
        gaussians.num_objects, state["weight"].shape[0], kernel_size=1
    ).cuda()
    classifier.load_state_dict(state)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    _, full_root, removed_root = render_roots(args)
    full_ply, removed_ply = model_files(args)
    writer = FrameWriter(
        full_root,
        args.backend,
        cameras,
        input_records(args, full_ply, "full", cfg_path, classifier_path),
        args.alpha_min,
    )
    render_set(
        str(args.model_path),
        "virtual",
        args.iteration,
        views,
        gaussians,
        pipeline,
        background,
        classifier,
        frame_writer=writer,
    )
    writer.finish()
    gaussians.load_ply(str(removed_ply))
    writer = FrameWriter(
        removed_root,
        args.backend,
        cameras,
        input_records(args, removed_ply, "all_selected", cfg_path, classifier_path),
        args.alpha_min,
    )
    render_set_removal_stage(
        str(args.model_path),
        "virtual",
        f"_object_removal/iteration_{args.iteration}",
        views,
        gaussians,
        pipeline,
        background,
        classifier,
        frame_writer=writer,
    )
    writer.finish()
    # Preserve the original optional target-only diagnostic views.
    target_ply = (
        removed_ply.parent.with_name(removed_ply.parent.name + "_removal_target")
        / "point_cloud.ply"
    )
    if target_ply.is_file():
        gaussians.load_ply(str(target_ply))
        render_set(
            str(args.model_path),
            "virtual",
            f"object_removal/iteration_{args.iteration}_removal_target",
            views,
            gaussians,
            pipeline,
            background,
            classifier,
        )


def pgsr(args):
    import torch
    from omegaconf import OmegaConf
    from source.vendor import bootstrap_gaussian_splatting
    from source.renderers.pgsr import PGSRRenderer

    bootstrap_gaussian_splatting()
    from scene import GaussianModel

    cfg_path = args.edgs_model_path / "config.yaml"
    config = OmegaConf.load(cfg_path)
    if OmegaConf.select(config, "gs.renderer.backend") != "pgsr":
        raise ValueError(
            "edgs-pgsr requires a model config with gs.renderer.backend=pgsr"
        )
    module, cameras = load_cameras(args)
    first = cameras["cameras"][0]
    base = SimpleNamespace(
        world_view_transform=torch.eye(4, device="cuda"),
        image_width=first["image_width"],
        image_height=first["image_height"],
    )
    views = module.virtual_views_from_manifest(base, cameras)
    background = torch.tensor(
        [1, 1, 1] if config.gs.dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    # These are evaluation settings; this rasterizer does not implement AA/debug.
    pipeline = SimpleNamespace(
        convert_SHs_python=bool(config.gs.pipe.get("convert_SHs_python", False)),
        compute_cov3D_python=bool(config.gs.pipe.get("compute_cov3D_python", False)),
        debug=False,
        antialiasing=False,
    )
    renderer = PGSRRenderer()
    _, full_root, removed_root = render_roots(args)
    full_ply, removed_ply = model_files(args)
    gaussians = GaussianModel(int(config.gs.sh_degree))
    for ply, root, variant in (
        (full_ply, full_root, "full"),
        (removed_ply, removed_root, "all_selected"),
    ):
        gaussians.load_ply(str(ply))
        writer = FrameWriter(
            root,
            args.backend,
            cameras,
            input_records(args, ply, variant, cfg_path),
            args.alpha_min,
        )
        for view in views:
            package = renderer.render(
                view, gaussians, pipeline, background, return_plane=True
            )
            writer.write(view, package)
        writer.finish()


ADAPTERS = {"inpaint360gs": native, "edgs-pgsr": pgsr}


def main():
    args = parser().parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("virtual rendering requires CUDA")
    with torch.no_grad():
        ADAPTERS[args.backend](args)


if __name__ == "__main__":
    main()
