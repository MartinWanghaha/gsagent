from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


GSAGENT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = GSAGENT_ROOT / "scripts" / "semantic_gaussian_wrapping" / "run_ablation_matrix_mipnerf360.py"
SPEC = importlib.util.spec_from_file_location("sgw_ablation_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_ablation_matrix_contains_control_and_factorized_full_variants() -> None:
    assert tuple(runner.VARIANT_CONFIGS) == (
        "rgb_only",
        "semantic_render_only",
        "full",
        "full_no_mesh_feedback",
        "full_no_surface_topology",
        "full_no_confidence_propagation",
        "full_no_expert_certainty",
        "full_no_prune_replace",
    )


def test_fresh_ablation_command_is_isolated_and_explicit(tmp_path) -> None:
    args = runner.parse_args(
        [
            "--scene",
            "counter",
            "--variant",
            "full",
            "--steps",
            "train",
            "--output-root",
            str(tmp_path / "output"),
            "--sgw-root",
            str(tmp_path / "sgw"),
        ]
    )
    command = runner.build_train_command(args, "full", "counter")
    assert "--config" in command
    assert str(tmp_path / "sgw" / "configs" / "full.yaml") in command
    assert str(tmp_path / "output" / "full") in command
    assert "--eval" in command
    assert "--resume" not in command


def test_resume_ablation_command_never_replays_fresh_configuration(tmp_path) -> None:
    args = runner.parse_args(
        [
            "--scene",
            "counter",
            "--variant",
            "full_no_mesh_feedback",
            "--steps",
            "train",
            "--output-root",
            str(tmp_path),
            "--resume",
            "--iterations",
            "40000",
            "--set",
            "logging.log_interval=50",
        ]
    )
    command = runner.build_train_command(args, "full_no_mesh_feedback", "counter")
    assert "--resume" in command
    assert "--iterations" in command and "40000" in command
    assert "--config" not in command
    assert "--resolution" not in command
    assert "--eval" not in command
    assert "--semantic-path" not in command


def test_mesh_stage_uses_public_region_wrapping_launcher(tmp_path) -> None:
    args = runner.parse_args(
        [
            "--scene",
            "counter",
            "--variant",
            "full",
            "--steps",
            "mesh",
            "--output-root",
            str(tmp_path / "output"),
            "--sgw-root",
            str(tmp_path / "sgw"),
            "--mesh-max-gaussians",
            "125000",
            "--mesh-max-chart-gaussians",
            "10000",
            "--mesh-view-stride",
            "2",
            "--mesh-camera-scale",
            "0.5",
            "--mesh-target-faces",
            "100000",
            "--gpu",
            "0",
            "--skip-mesh-metrics",
        ]
    )
    root = runner.model_path(args, "full", "counter")
    command, output, metrics = runner.build_mesh_command(
        args,
        root,
        30_000,
        "counter",
        "full",
    )

    assert command[:2] == [runner.sys.executable, str(runner.MESH_LAUNCHER)]
    assert command[2] == str(root)
    assert command[command.index("--sgw-root") + 1] == str((tmp_path / "sgw").resolve())
    assert command[command.index("--max-gaussians") + 1] == "125000"
    assert command[command.index("--max-chart-gaussians") + 1] == "10000"
    assert command[command.index("--view-stride") + 1] == "2"
    assert command[command.index("--camera-scale") + 1] == "0.5"
    assert command[command.index("--target-faces") + 1] == "100000"
    assert command[command.index("--gpu") + 1] == "0"
    assert "--force" in command
    assert "--resolution" not in command
    assert "extract_mesh.py" not in command
    assert output == root / "mesh" / "iteration_30000" / "semantic_surface.ply"
    assert metrics is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("--mesh-max-gaussians", "3"),
        ("--mesh-max-chart-gaussians", "3"),
        ("--mesh-view-stride", "0"),
        ("--mesh-camera-scale", "0"),
        ("--mesh-camera-scale", "nan"),
        ("--mesh-target-faces", "0"),
        ("--iteration", "0"),
    ],
)
def test_invalid_region_wrapping_controls_are_rejected(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--scene",
                "counter",
                "--variant",
                "full",
                "--steps",
                "mesh",
                "--skip-mesh-metrics",
                option,
                value,
            ]
        )


def test_dry_run_has_no_filesystem_side_effects(tmp_path, capsys) -> None:
    output = tmp_path / "matrix"
    return_code = runner.main(
        [
            "--scene",
            "counter",
            "--variant",
            "rgb_only,full",
            "--steps",
            "train",
            "--output-root",
            str(output),
            "--dry-run",
        ]
    )
    assert return_code == 0
    assert not output.exists()
    printed = capsys.readouterr().out
    assert "rgb_only.yaml" in printed
    assert "full.yaml" in printed


def test_resume_resolves_evaluation_iteration_after_training(tmp_path, monkeypatch) -> None:
    run = tmp_path / "full" / "counter"
    run.mkdir(parents=True)
    (run / "chkpnt24000.pth").touch()
    commands = []

    def fake_run(command, _cwd, _environment, _dry_run):
        commands.append(command)
        if str(runner.TRAIN_LAUNCHER) in command:
            (run / "chkpnt30000.pth").touch()
        return 0

    monkeypatch.setattr(runner, "_run", fake_run)
    assert (
        runner.main(
            [
                "--scene",
                "counter",
                "--variant",
                "full",
                "--steps",
                "train,render",
                "--output-root",
                str(tmp_path),
                "--resume",
            ]
        )
        == 0
    )
    render_command = next(command for command in commands if "render.py" in command)
    assert render_command[render_command.index("--iteration") + 1] == "30000"


def test_missing_mesh_reference_fails_before_any_stage_runs(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(runner, "_run", lambda *args: calls.append(args) or 0)
    assert (
        runner.main(
            [
                "--scene",
                "counter",
                "--variant",
                "full",
                "--steps",
                "train,mesh",
                "--output-root",
                str(tmp_path),
                "--mesh-reference",
                str(tmp_path / "missing-{scene}.ply"),
            ]
        )
        == 2
    )
    assert calls == []
