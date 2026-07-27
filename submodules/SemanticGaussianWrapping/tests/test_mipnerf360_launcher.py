from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from utils.config_utils import load_config, validate_config


GSAGENT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = GSAGENT_ROOT / "scripts" / "semantic_gaussian_wrapping" / "train_semantic_gaussian_wrapping_mipnerf360.py"
SPEC = importlib.util.spec_from_file_location("sgw_mipnerf360_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _args(**overrides):
    values = {
        "iterations": 30_000,
        "images": "images",
        "resolution": -1,
        "data_device": "cpu",
        "eval": True,
        "holdout": 8,
        "max_gaussians": None,
        "overrides": [],
        "device": "cuda",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fresh_command_translates_wrapper_flags_to_config_overrides(tmp_path) -> None:
    command = launcher.build_command(
        tmp_path / "data" / "counter",
        tmp_path / "output" / "counter",
        tmp_path / "full.yaml",
        _args(max_gaussians=3_000_000),
        "sam_mask",
        "",
        "",
        ["--quiet"],
    )
    joined = " ".join(command)

    assert "--config" in command and "--checkpoint" not in command
    assert "optimization.iterations=30000" in command
    assert "data.resolution=-1" in command
    assert "data.eval=true" in command
    assert 'data.semantic_confidence=""' in command
    assert 'data.semantic_boundary=""' in command
    assert "density.max_gaussians=3000000" in command
    assert command[-1] == "--quiet"
    assert "--iterations" not in joined and " --eval" not in joined
    expressions = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--set"]
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "full.yaml", expressions)
    validate_config(config)
    assert config["data"]["resolution"] == -1
    assert config["data"]["eval"] is True


def test_resume_uses_checkpoint_config_and_only_resume_safe_overrides(tmp_path) -> None:
    checkpoint = tmp_path / "output" / "counter" / "chkpnt12000.pth"
    command = launcher.build_command(
        tmp_path / "data" / "counter",
        checkpoint.parent,
        tmp_path / "full.yaml",
        _args(
            overrides=[
                "logging.log_interval=50",
                "semantic.region_decode_chunk_size=16384",
                "surface.support_routing_query_chunk=4096",
                "surface.scipy_workers=2",
                "surface.mesh_feedback_scipy_workers=1",
            ]
        ),
        "sam_mask",
        "",
        "",
        [],
        checkpoint,
    )

    assert "--checkpoint" in command and str(checkpoint) in command
    assert "--config" not in command
    assert "data.resolution=-1" not in command
    assert "optimization.iterations=30000" in command
    assert "logging.log_interval=50" in command
    assert "semantic.region_decode_chunk_size=16384" in command
    assert "surface.support_routing_query_chunk=4096" in command
    assert "surface.scipy_workers=2" in command
    assert "surface.mesh_feedback_scipy_workers=1" in command


def test_resume_rejects_objective_changes(tmp_path) -> None:
    with pytest.raises(ValueError, match="resume cannot change"):
        launcher.build_command(
            tmp_path / "data" / "counter",
            tmp_path / "output" / "counter",
            tmp_path / "full.yaml",
            _args(overrides=["loss.lambda_semantic=0.5"]),
            "sam_mask",
            "",
            "",
            [],
            tmp_path / "chkpnt12000.pth",
        )
    with pytest.raises(ValueError, match="resume cannot change"):
        launcher.build_command(
            tmp_path / "data" / "counter",
            tmp_path / "output" / "counter",
            tmp_path / "full.yaml",
            _args(overrides=["surface.support_candidate_budget=512"]),
            "sam_mask",
            "",
            "",
            [],
            tmp_path / "chkpnt12000.pth",
        )


def test_resume_reads_authoritative_observation_layout(tmp_path) -> None:
    model = tmp_path / "counter"
    model.mkdir()
    checkpoint = model / "chkpnt12000.pth"
    torch.save(
        {
            "version": 3,
            "iteration": 12_000,
            "config": {
                "data": {
                    "images": "images_4",
                    "semantic_path": "entityseg_mask",
                    "semantic_confidence": "confidence",
                    "semantic_boundary": "boundary",
                }
            },
        },
        checkpoint,
    )
    # The mutable convenience copy is deliberately stale; resume owns the
    # immutable configuration embedded in the selected checkpoint.
    (model / "config.yaml").write_text(
        "data:\n  images: wrong_images\n  semantic_path: wrong_masks\n",
        encoding="utf8",
    )

    config, iteration = launcher.load_checkpoint_configuration(checkpoint)
    assert iteration == 12_000
    assert launcher.resume_data_options(config, checkpoint) == (
        "images_4",
        "entityseg_mask",
        "confidence",
        "boundary",
    )


def test_launcher_checkpoint_metadata_prefers_mmap(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "chkpnt12000.pth"
    state = {"version": 3, "iteration": 12_000, "config": {"data": {}}}
    calls = []

    def recording_load(path, **kwargs):
        calls.append((path, kwargs))
        return state

    monkeypatch.setattr(torch, "load", recording_load)
    config, iteration = launcher.load_checkpoint_configuration(checkpoint)

    assert config is state["config"]
    assert iteration == 12_000
    assert calls == [
        (
            checkpoint,
            {"mmap": True, "map_location": "cpu", "weights_only": True},
        )
    ]


@pytest.mark.parametrize(
    ("version", "message"),
    [(2, "region-conditioned training schema"), (4, "newer training schema")],
)
def test_launcher_rejects_non_v3_checkpoint_metadata(
    monkeypatch,
    tmp_path,
    version,
    message,
) -> None:
    checkpoint = tmp_path / "chkpnt12000.pth"
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "version": version,
            "iteration": 12_000,
            "config": {"data": {}},
        },
    )

    with pytest.raises(ValueError, match=message):
        launcher.load_checkpoint_configuration(checkpoint)


@pytest.mark.parametrize("version", ["3", 3.0, None])
def test_launcher_rejects_non_integer_checkpoint_schema(
    monkeypatch, tmp_path, version
) -> None:
    checkpoint = tmp_path / "chkpnt12000.pth"
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "version": version,
            "iteration": 12_000,
            "config": {"data": {}},
        },
    )
    with pytest.raises(ValueError, match="invalid training schema"):
        launcher.load_checkpoint_configuration(checkpoint)


def test_semantic_validation_requires_info_and_complete_stems(tmp_path) -> None:
    semantic = tmp_path / "sam_mask"
    semantic.mkdir()
    (semantic / "first.png").touch()

    with pytest.raises(launcher.ObservationValidationError, match="metadata"):
        launcher.validate_observations(
            semantic,
            {"first", "second"},
            label="semantic",
            require_info=True,
        )

    (semantic / "info.json").write_text('{"num_mask": 2}\n', encoding="utf8")
    with pytest.raises(launcher.ObservationValidationError, match="missing 1/2"):
        launcher.validate_observations(
            semantic,
            {"first", "second"},
            label="semantic",
            require_info=True,
        )

    (semantic / "second.npy").touch()
    assert launcher.validate_observations(
        semantic,
        {"first", "second"},
        label="semantic",
        require_info=True,
    ) == (2, 2)


def test_semantic_validation_rejects_invalid_metadata_and_shape(tmp_path) -> None:
    semantic = tmp_path / "sam_mask"
    semantic.mkdir()
    Image.new("L", (5, 3), color=2).save(semantic / "first.png")
    (semantic / "info.json").write_text('{"num_mask": "many"}\n', encoding="utf8")

    with pytest.raises(launcher.ObservationValidationError, match="positive integer"):
        launcher.validate_observations(
            semantic,
            {"first"},
            label="semantic",
            require_info=True,
            expected_shapes={"first": (3, 5)},
        )

    (semantic / "info.json").write_text('{"num_mask": 2}\n', encoding="utf8")
    with pytest.raises(launcher.ObservationValidationError, match="unaligned"):
        launcher.validate_observations(
            semantic,
            {"first"},
            label="semantic",
            require_info=True,
            expected_shapes={"first": (4, 5)},
        )

    semantic_native_shapes = {}
    assert launcher.validate_observations(
        semantic,
        {"first"},
        label="semantic",
        require_info=True,
        expected_shapes={"first": (6, 10)},
        decoded_shapes=semantic_native_shapes,
    ) == (1, 1)
    confidence = tmp_path / "confidence"
    confidence.mkdir()
    Image.new("L", (10, 6), color=255).save(confidence / "first.png")
    with pytest.raises(launcher.ObservationValidationError, match="semantic native shape"):
        launcher.validate_observations(
            confidence,
            {"first"},
            label="confidence",
            require_info=False,
            expected_shapes={"first": (6, 10)},
            native_reference_shapes=semantic_native_shapes,
        )

    np.save(semantic / "two_channel.npy", np.zeros((3, 5, 2), dtype=np.uint8))
    with pytest.raises(launcher.ObservationValidationError, match="two channels"):
        launcher.validate_observations(
            semantic,
            {"two_channel"},
            label="semantic",
            require_info=True,
            expected_shapes={"two_channel": (3, 5)},
        )

    Image.new("L", (5, 3), color=0).save(semantic / "background.png")
    with pytest.raises(launcher.ObservationValidationError, match="no foreground"):
        launcher.validate_observations(
            semantic,
            {"background"},
            label="semantic",
            require_info=True,
            expected_shapes={"background": (3, 5)},
        )


def test_fresh_configuration_fails_before_native_train_for_invalid_schedule() -> None:
    workdir = Path(__file__).resolve().parents[1]
    config_path = workdir / "configs" / "full.yaml"
    invalid = launcher.fresh_overrides(_args(iterations=12_000), "sam_mask", "", "")
    with pytest.raises(ValueError, match="phase boundaries"):
        launcher.validate_fresh_configuration(workdir, config_path, invalid)

    valid = launcher.fresh_overrides(
        _args(
            iterations=12_000,
            overrides=[
                "phases.semantic_from=3000",
                "phases.joint_from=6000",
                "phases.surface_from=10000",
                "surface.topology_from=10000",
                "surface.topology_until=12000",
            ],
        ),
        "sam_mask",
        "",
        "",
    )
    resolved = launcher.validate_fresh_configuration(workdir, config_path, valid)
    assert resolved["optimization"]["iterations"] == 12_000


def test_resume_configuration_validates_extended_curriculum() -> None:
    workdir = Path(__file__).resolve().parents[1]
    config = load_config(workdir / "configs" / "full.yaml")

    with pytest.raises(ValueError, match="phase boundaries"):
        launcher.validate_resume_configuration(
            workdir,
            config,
            7_000,
            launcher.resume_overrides(_args(iterations=12_000)),
        )

    resolved = launcher.validate_resume_configuration(
        workdir,
        config,
        24_000,
        launcher.resume_overrides(
            _args(
                iterations=30_000,
                overrides=[
                    "semantic.region_decode_chunk_size=16384",
                    "surface.support_routing_query_chunk=4096",
                    "surface.scipy_workers=2",
                    "surface.mesh_feedback_scipy_workers=1",
                ],
            )
        ),
    )
    assert resolved["optimization"]["iterations"] == 30_000
    assert resolved["semantic"]["region_decode_chunk_size"] == 16_384
    assert resolved["surface"]["support_routing_query_chunk"] == 4_096
    assert resolved["surface"]["scipy_workers"] == 2
    assert resolved["surface"]["mesh_feedback_scipy_workers"] == 1


def test_resume_without_iterations_preserves_checkpoint_target(tmp_path) -> None:
    checkpoint = tmp_path / "chkpnt20000.pth"
    config = {"optimization": {"iterations": 40_000}}

    implicit, _ = launcher.parse_args(["--resume"])
    assert launcher.resume_target_iteration(implicit, config, checkpoint) == 40_000

    explicit, _ = launcher.parse_args(["--resume", "--iterations=50000"])
    assert launcher.resume_target_iteration(explicit, config, checkpoint) == 50_000


def test_wrapper_controlled_set_and_misplaced_native_flags_fail_fast() -> None:
    with pytest.raises(SystemExit):
        launcher.parse_args(["--set", "data.semantic_path=wrong_masks"])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--set", 'data={"semantic_path":"wrong_masks"}'])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--unknown", "value", "--", "--quiet"])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--resume", "--semantic-path", "sam_mask"])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--resume", "-r4"])

    _, native = launcher.parse_args(["--", "--quiet"])
    assert native == ["--quiet"]


def test_resume_run_requires_resume_and_exactly_one_concrete_scene() -> None:
    with pytest.raises(SystemExit):
        launcher.parse_args(
            ["--scene", "counter", "--resume-run", "counter_rerun_002"]
        )
    with pytest.raises(SystemExit):
        launcher.parse_args(["--resume", "--resume-run", "counter_rerun_002"])
    with pytest.raises(SystemExit):
        launcher.parse_args(
            [
                "--resume",
                "--scene",
                "counter,garden",
                "--resume-run",
                "counter_rerun_002",
            ]
        )
    with pytest.raises(SystemExit):
        launcher.parse_args(
            ["--resume", "--scene", "all", "--resume-run", "counter_rerun_002"]
        )

    args, _ = launcher.parse_args(
        [
            "--resume",
            "--scene",
            "counter",
            "--resume-run",
            "counter_rerun_002",
        ]
    )
    assert args.resume_run == Path("counter_rerun_002")


def test_resume_run_resolves_relative_to_output_root_or_as_absolute(tmp_path) -> None:
    output_root = tmp_path / "output"
    run = output_root / "counter_rerun_002"
    run.mkdir(parents=True)

    assert launcher.resolve_resume_run(Path("counter_rerun_002"), output_root) == run
    assert launcher.resolve_resume_run(run, output_root) == run
    with pytest.raises(ValueError, match="remain below"):
        launcher.resolve_resume_run(Path("../outside"), output_root)


def test_force_allocates_an_isolated_output_directory(tmp_path) -> None:
    base = tmp_path / "counter"
    base.mkdir()
    (base / "chkpnt30000.pth").touch()
    first_rerun = tmp_path / "counter_rerun_001"
    first_rerun.mkdir()
    (first_rerun / "config.yaml").touch()

    assert launcher.select_model_path(base, force=False) == base
    assert launcher.select_model_path(base, force=True) == tmp_path / "counter_rerun_002"

    empty = tmp_path / "room"
    empty.mkdir()
    assert launcher.select_model_path(empty, force=True) == tmp_path / "room_rerun_001"


def test_resume_scene_discovery_does_not_require_wrapper_image_directory(tmp_path) -> None:
    sparse = tmp_path / "custom" / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).touch()

    assert launcher.discover_scenes(tmp_path, images=None) == ["custom"]
    assert launcher.discover_scenes(tmp_path, images="images") == []


def test_resume_dry_run_uses_checkpoint_owned_custom_images(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "data" / "counter"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).touch()
    custom_images = source / "custom_images"
    custom_images.mkdir()
    Image.new("RGB", (5, 3)).save(custom_images / "view.png")
    masks = source / "sam_mask"
    masks.mkdir()
    Image.new("L", (10, 6), color=2).save(masks / "view.png")
    (masks / "info.json").write_text('{"num_mask": 2}\n', encoding="utf8")

    model = tmp_path / "output" / "counter_rerun_002"
    model.mkdir(parents=True)
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "full.yaml")
    config["data"]["images"] = "custom_images"
    config["runtime"] = {"source_path": str(source.resolve())}
    torch.save(
        {"version": 3, "iteration": 12_000, "config": config},
        model / "chkpnt12000.pth",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--scene",
            "counter",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--resume",
            "--resume-run",
            "counter_rerun_002",
            "--dry-run",
        ],
    )

    assert launcher.main() == 0
    output = capsys.readouterr().out
    assert "[resume] counter: iteration 12000 -> 30000" in output
    assert "--checkpoint" in output
    assert str(model) in output


def test_forced_dry_run_does_not_reserve_rerun_directory(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "data" / "custom"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).touch()
    images = source / "images"
    images.mkdir()
    Image.new("RGB", (5, 3)).save(images / "view.png")
    masks = source / "sam_mask"
    masks.mkdir()
    Image.new("L", (5, 3), color=2).save(masks / "view.png")
    (masks / "info.json").write_text('{"num_mask": 2}\n', encoding="utf8")

    existing = tmp_path / "output" / "custom"
    existing.mkdir(parents=True)
    (existing / "config.yaml").touch()
    rerun = tmp_path / "output" / "custom_rerun_001"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--scene",
            "custom",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
            "--force",
            "--dry-run",
        ],
    )

    assert launcher.main() == 0
    assert not rerun.exists()
    assert str(rerun) in capsys.readouterr().out


def test_latest_checkpoint_never_selects_newer_than_target(tmp_path) -> None:
    for iteration in (7_000, 12_000, 30_000):
        (tmp_path / f"chkpnt{iteration}.pth").touch()

    path, iteration, latest = launcher.latest_checkpoint(tmp_path, 20_000)

    assert path == tmp_path / "chkpnt12000.pth"
    assert iteration == 12_000
    assert latest == 30_000
