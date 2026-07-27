"""Static launcher tests that do not require a CUDA device."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from _common import model_directories, run_directory, validate_masks
from extract_gaussian_wrapping_gaga_mesh import (
    extraction_command,
    generated_mesh_path,
    main as mesh_main,
)
from train_gaussian_wrapping_gaga_mipnerf360 import (
    geometry_command,
    lift_command,
    render_command,
    render_outputs_complete,
)


def training_args() -> Namespace:
    return Namespace(
        images="images",
        geometry_iterations=30_000,
        semantic_iterations=10_000,
        resolution=-1,
        rasterizer="radegs",
        data_device="cpu",
        port=6009,
        max_gaussians=6_000_000,
        eval=True,
        depth_order=False,
        depth_order_config=None,
        no_paper_defaults=False,
        semantic_num_classes=None,
        semantic_lr=2.5e-3,
        semantic_head_lr=5e-4,
        lambda_semantic=1.0,
        lambda_semantic_3d=0.01,
        semantic_3d_interval=10,
        semantic_3d_samples=10_000,
        semantic_3d_neighbors=5,
        allow_missing_masks=False,
        render_after_train=True,
        render_resolution=2,
        render_output_profile="images",
        render_class_chunk_size=32_768,
    )


def extraction_args() -> Namespace:
    return Namespace(
        n_pivots=2,
        isosurface_value=0.0,
        n_binary_steps=10,
        dtype="int32",
        data_device="cpu",
        resolution=None,
        sdf_batch_size=None,
        mtet_on_cpu=False,
        no_postprocess=False,
    )


def test_output_layout_separates_modes(tmp_path):
    two_stage = run_directory(tmp_path, "counter", "two-stage", "entityseg")
    joint = run_directory(tmp_path, "counter", "joint", "entityseg")
    geometry, semantics = model_directories(two_stage, "two-stage")
    joint_geometry, joint_semantics = model_directories(joint, "joint")
    assert geometry != semantics
    assert joint_geometry == joint_semantics
    assert two_stage != joint


def test_joint_command_contains_native_semantic_training():
    args = training_args()
    command = geometry_command(
        Path("/data/counter"),
        Path("/output/model"),
        0,
        args,
        [],
        joint=True,
        mask_dir=Path("/data/counter/entityseg_mask"),
        inferred_num_classes=484,
    )
    assert "--semantic_masks" in command
    assert command[command.index("--semantic_num_classes") + 1] == "484"
    assert "--lambda_semantic" in command
    assert "--lambda_semantic_3d" in command


def test_two_stage_commands_keep_geometry_and_semantics_separate():
    args = training_args()
    geometry = geometry_command(
        Path("/data/counter"),
        Path("/output/geometry"),
        0,
        args,
        [],
        joint=False,
        mask_dir=Path("/data/counter/entityseg_mask"),
        inferred_num_classes=484,
    )
    semantics = lift_command(
        Path("/data/counter"),
        Path("/output/geometry"),
        Path("/output/semantic"),
        Path("/data/counter/entityseg_mask"),
        484,
        args,
    )
    assert "--semantic_masks" not in geometry
    assert "semantic_lift.py" in semantics
    assert semantics[semantics.index("--num_classes") + 1] == "484"


def test_post_training_render_command_matches_gaga_layout_contract():
    args = training_args()
    command = render_command(
        Path("/data/counter"),
        Path("/output/semantic"),
        Path("/output/semantic/semantic/semantic_chkpnt10000.pth"),
        Path("/output/run"),
        Path("/data/counter/entityseg_mask"),
        10_000,
        args,
    )
    assert "semantic_render.py" in command
    assert command[command.index("--split") + 1] == "all"
    assert command[command.index("--output") + 1] == "/output/run"
    assert command[command.index("--load_iteration") + 1] == "10000"
    assert command[command.index("--output_profile") + 1] == "images"
    assert command[command.index("-r") + 1] == "2"


def test_completed_gaga_compatible_renders_are_detected(tmp_path):
    iteration = 10_000
    counts = {"train": 2, "test": 1}
    for split, count in counts.items():
        root = tmp_path / split / f"ours_{iteration}"
        directories = [
            "renders",
            "gt",
            "objects_feature16",
            "objects_pred",
            "objects_test",
        ]
        if split == "train":
            directories.extend(("gt_objects", "gt_objects_color"))
        for directory in directories:
            path = root / directory
            path.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (path / f"{index}.png").touch()
    (tmp_path / "render_manifest.json").write_text(
        (
            '{"iteration": 10000, "resolution": 2, '
            '"output_profile": "images", '
            '"splits": {"train": {"count": 2}, "test": {"count": 1}}}'
        ),
        encoding="utf-8",
    )
    assert render_outputs_complete(
        tmp_path,
        iteration=iteration,
        resolution=2,
        output_profile="images",
        eval_mode=True,
    )


def test_mesh_presets_match_gaussian_wrapping():
    args = extraction_args()
    ours = extraction_command(
        Path("/model"),
        30_000,
        Path("/data/counter"),
        "ours",
        args,
        [],
    )
    radegs = extraction_command(
        Path("/model"),
        30_000,
        Path("/data/counter"),
        "radegs",
        args,
        [],
    )
    assert "--filter_large_edges" in ours
    assert "--use_searched_pivots" in radegs
    assert generated_mesh_path(
        Path("/model"), "ours", 2, 0.0, True
    ).name == "mesh_ours_2pivots_post.ply"
    assert generated_mesh_path(
        Path("/model"), "radegs", 2, 0.0, True
    ).name == "mesh_exact_computation_2pivots_searched_post.ply"


def test_mask_validation_uses_gaga_info_and_camera_stems(tmp_path):
    scene = tmp_path / "counter"
    images = scene / "images"
    masks = scene / "entityseg_mask"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "frame.jpg").write_bytes(b"fixture")
    (masks / "frame.png").write_bytes(b"fixture")
    (masks / "info.json").write_text('{"num_mask": 7}', encoding="utf-8")
    mask_dir, num_classes = validate_masks(
        scene,
        "entityseg",
        "images",
        allow_missing=False,
    )
    assert mask_dir == masks
    assert num_classes == 8


def test_mesh_launcher_dry_run_resolves_two_stage_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
):
    data_root = tmp_path / "data"
    scene = data_root / "counter"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).touch()

    output_root = tmp_path / "outputs"
    run = output_root / "counter" / "two-stage" / "entityseg"
    geometry_ply = (
        run
        / "geometry"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    semantic_ply = (
        run
        / "semantic"
        / "point_cloud"
        / "iteration_10000"
        / "point_cloud.ply"
    )
    semantic_checkpoint = (
        run / "semantic" / "semantic" / "semantic_chkpnt10000.pth"
    )
    for path in (geometry_ply, semantic_ply, semantic_checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (run / "run_manifest.json").write_text(
        '{"rasterizer": "radegs"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_gaussian_wrapping_gaga_mesh.py",
            "--scene",
            "counter",
            "--mode",
            "two-stage",
            "--mask-method",
            "entityseg",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )
    assert mesh_main() == 0
    output = capsys.readouterr().out
    assert "[mesh:counter:extract]" in output
    assert "[mesh:counter:semantic]" in output
    assert "semantic_chkpnt10000.pth" in output
