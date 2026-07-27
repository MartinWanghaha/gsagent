"""Lift associated Gaga masks into a trained Gaussian Wrapping model."""

from __future__ import annotations

import os
import random
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
SUBMODULES_DIR = os.path.join(ROOT_DIR, "submodules")
sys.path.extend((ROOT_DIR, SUBMODULES_DIR))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from semantic import (
    GagaObservationStore,
    SemanticHead,
    save_semantic_checkpoint,
    semantic_cross_entropy,
    spatial_consistency_loss,
)
from utils.general_utils import safe_state


def select_renderer(name: str):
    if name == "radegs":
        from gaussian_renderer.radegs import render_radegs

        return render_radegs
    if name == "ours":
        from gaussian_renderer.ours import render_ours

        return render_ours
    raise ValueError(f"Unknown renderer: {name}")


def save_state(
    output_dir: Path,
    iteration: int,
    gaussians: GaussianModel,
    head: SemanticHead,
    optimizer,
    num_classes: int,
    renderer: str,
    mask_dir: str,
) -> None:
    iteration_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(iteration_dir / "point_cloud.ply"))
    save_semantic_checkpoint(
        output_dir / "semantic",
        head=head,
        gaussian_model=gaussians,
        iteration=iteration,
        num_classes=num_classes,
        renderer=renderer,
        optimizer=optimizer,
        metadata={"mask_dir": os.path.abspath(mask_dir), "mode": "lift"},
    )


def lift(dataset, pipe, args) -> None:
    gaussians = GaussianModel(
        dataset.sh_degree,
        semantic_dim=16,
        use_unbounded_opacity=dataset.use_unbounded_opacity,
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
    )
    if not gaussians.use_semantic_features or gaussians._semantic_features.numel() == 0:
        gaussians.initialize_semantic_features(16)

    cameras = scene.getTrainCameras().copy()
    observations = GagaObservationStore(
        args.semantic_masks,
        require_all=not args.allow_missing_masks,
    )
    num_classes = args.num_classes or observations.validate_cameras(cameras)
    if num_classes < 2:
        raise ValueError("At least background and one associated Gaga instance are required")
    head = SemanticHead(16, num_classes).cuda()
    gaussians.semantic_training_setup(args.semantic_lr)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    render = select_renderer(args.rasterizer)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    output_dir = Path(args.semantic_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config = Path(dataset.model_path) / "cfg_args"
    if source_config.is_file():
        shutil.copy2(source_config, output_dir / "cfg_args")
    camera_stack = []
    progress = tqdm(range(1, args.semantic_iterations + 1), desc="Semantic lift")
    for iteration in progress:
        if not camera_stack:
            camera_stack = cameras.copy()
        camera = camera_stack.pop(random.randrange(len(camera_stack)))
        observation = observations.load(
            camera.image_name,
            camera.image_height,
            camera.image_width,
        ).to("cuda")

        render_pkg = render(
            camera,
            gaussians,
            pipe,
            background,
            render_semantics=True,
            require_coord=False,
            require_depth=False,
        )
        logits = head(render_pkg["semantic_features"])
        loss_2d = semantic_cross_entropy(
            logits,
            observation.labels,
            confidence=observation.confidence,
            ignore_index=observations.ignore_label,
        )
        loss_3d = logits.sum() * 0.0
        if args.lambda_semantic_3d > 0 and iteration % args.semantic_3d_interval == 0:
            loss_3d = spatial_consistency_loss(
                gaussians.get_semantic_features,
                gaussians.get_xyz,
                sample_size=args.semantic_3d_samples,
                neighbors=args.semantic_3d_neighbors,
            )
        loss = loss_2d + args.lambda_semantic_3d * loss_3d

        gaussians.optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gaussians.optimizer.step()
        head_optimizer.step()

        if iteration % 10 == 0:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_2d.item():.4f}",
            )
        if iteration in args.save_iterations or iteration == args.semantic_iterations:
            save_state(
                output_dir,
                iteration,
                gaussians,
                head,
                {
                    "gaussian": gaussians.optimizer.state_dict(),
                    "head": head_optimizer.state_dict(),
                },
                num_classes,
                args.rasterizer,
                args.semantic_masks,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Wrapping Gaga semantic lift")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--semantic_masks", required=True)
    parser.add_argument("--semantic_output", required=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--semantic_iterations", type=int, default=10_000)
    parser.add_argument("--semantic_lr", type=float, default=2.5e-3)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--rasterizer", choices=("radegs", "ours"), default="radegs")
    parser.add_argument("--lambda_semantic_3d", type=float, default=0.0)
    parser.add_argument("--semantic_3d_interval", type=int, default=10)
    parser.add_argument("--semantic_3d_samples", type=int, default=10_000)
    parser.add_argument("--semantic_3d_neighbors", type=int, default=5)
    parser.add_argument("--allow_missing_masks", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    arguments = get_combined_args(parser)
    safe_state(arguments.quiet)
    lift(model.extract(arguments), pipeline.extract(arguments), arguments)
