from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


GSAGENT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    GSAGENT_ROOT
    / "scripts"
    / "semantic_gaussian_wrapping"
    / "extract_semantic_gaussian_wrapping_mesh.py"
)
SPEC = importlib.util.spec_from_file_location("sgw_mesh_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _scene(root: Path, *iterations: int) -> Path:
    root.mkdir()
    (root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    for iteration in iterations:
        (root / f"chkpnt{iteration}.pth").touch()
    return root


def test_latest_iteration_requires_a_complete_checkpoint(tmp_path) -> None:
    scene = _scene(tmp_path / "counter", 7_000, 12_000)
    assert launcher.resolve_iteration(scene, -1) == 12_000
    assert launcher.resolve_iteration(scene, 7_000) == 7_000
    with pytest.raises(FileNotFoundError, match="12001"):
        launcher.resolve_iteration(scene, 12_001)
    with pytest.raises(FileNotFoundError, match="complete checkpoint"):
        launcher.resolve_iteration(tmp_path / "empty", -1)


def test_wrapper_builds_only_rcgw_options(tmp_path) -> None:
    scene = _scene(tmp_path / "counter", 12_000)
    reference = tmp_path / "reference.ply"
    reference.touch()
    metrics = tmp_path / "metrics.json"
    args = launcher.parse_args(
        [
            str(scene),
            "--iteration",
            "12000",
            "--max-gaussians",
            "125000",
            "--max-chart-gaussians",
            "8000",
            "--view-stride",
            "2",
            "--camera-scale",
            "0.5",
            "--target-faces",
            "100000",
            "--reference",
            str(reference),
            "--metric-threshold",
            "0.02",
            "--metric-samples",
            "50000",
            "--metrics-json",
            str(metrics),
            "--gpu",
            "0",
        ]
    )
    output = launcher.default_output_path(scene, 12_000)
    command = launcher.build_command(args, scene, 12_000, output)

    assert command[:2] == [launcher.sys.executable, "extract_mesh.py"]
    assert ["-m", str(scene)] == command[2:4]
    assert command[command.index("--max-gaussians") + 1] == "125000"
    assert command[command.index("--max-chart-gaussians") + 1] == "8000"
    assert command[command.index("--view-stride") + 1] == "2"
    assert command[command.index("--camera-scale") + 1] == "0.5"
    assert command[command.index("--target-faces") + 1] == "100000"
    assert command[command.index("--reference") + 1] == str(reference.resolve())
    assert command[command.index("--metrics-json") + 1] == str(metrics.resolve())
    for removed in (
        "--method",
        "--scalar",
        "--level",
        "--resolution",
        "--region-id",
        "--all-regions",
        "--query-chunk-size",
        "--scipy-workers",
        "--max-pivots",
    ):
        assert removed not in command


@pytest.mark.parametrize(
    "arguments",
    [
        ["--method", "marching_cubes"],
        ["--resolution", "256"],
        ["--all-regions"],
        ["--max-pivots", "100"],
        ["--", "--target-faces", "100"],
        ["--iteration", "0"],
        ["--max-gaussians", "3"],
        ["--max-chart-gaussians", "3"],
        ["--view-stride", "0"],
        ["--camera-scale", "0"],
        ["--camera-scale", "nan"],
        ["--target-faces", "0"],
        ["--metrics-json", "metrics.json"],
    ],
)
def test_wrapper_rejects_old_unknown_or_invalid_options(
    tmp_path,
    arguments: list[str],
) -> None:
    scene = _scene(tmp_path / "counter", 12_000)
    with pytest.raises(SystemExit):
        launcher.parse_args([str(scene), *arguments])


def test_unspecified_policy_options_are_left_to_native_defaults(tmp_path) -> None:
    scene = _scene(tmp_path / "counter", 12_000)
    args = launcher.parse_args([str(scene)])
    output = launcher.default_output_path(scene, 12_000)
    command = launcher.build_command(args, scene, 12_000, output)

    assert "--max-gaussians" not in command
    assert "--max-chart-gaussians" not in command
    assert "--view-stride" not in command
    assert "--camera-scale" not in command
    assert "--target-faces" not in command


def test_wrapper_routes_multiview_visible_high_precision_method(tmp_path) -> None:
    scene = _scene(tmp_path / "counter", 30_000)
    args = launcher.parse_args(
        [
            str(scene),
            "--iteration",
            "30000",
            "--method",
            "multiview-visible",
            "--device",
            "cpu",
        ]
    )
    output = launcher.default_output_path(
        scene,
        30_000,
        launcher.METHOD_MULTIVIEW,
    )
    command = launcher.build_command(args, scene, 30_000, output)

    assert command[:2] == [
        launcher.sys.executable,
        "extract_multiview_gaussian_mesh.py",
    ]
    assert command[command.index("--device") + 1] == "cpu"
    assert output.name == (
        "semantic_multiview_gaussian_wrapping_iteration_30000_high_precision.ply"
    )


def test_complete_output_requires_rcgw_v2_ply_and_nonempty_counts(tmp_path) -> None:
    output = tmp_path / "surface.ply"
    sidecar = launcher.manifest_path(output)
    output.write_bytes(b"ply\npayload")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "algorithm": "region_conditioned_gaussian_wrapping",
                "iteration": 12_000,
                "vertices": 3,
                "faces": 1,
            }
        ),
        encoding="utf-8",
    )

    assert launcher.output_is_complete(output, 12_000)
    assert not launcher.output_is_complete(output, 30_000)
    output.write_bytes(b"")
    assert not launcher.output_is_complete(output, 12_000)


def test_legacy_manifest_is_never_considered_complete(tmp_path) -> None:
    output = tmp_path / "surface.ply"
    output.write_bytes(b"ply\npayload")
    launcher.manifest_path(output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "semantic_visibility_pivots",
                "iteration": 12_000,
                "vertices": 3,
                "faces": 1,
            }
        ),
        encoding="utf-8",
    )

    assert not launcher.output_is_complete(output, 12_000)


def test_dry_run_validates_experiment_contract(tmp_path, capsys) -> None:
    scene = _scene(tmp_path / "counter", 12_000)
    sgw_root = tmp_path / "SemanticGaussianWrapping"
    sgw_root.mkdir()
    (sgw_root / "extract_mesh.py").touch()

    assert (
        launcher.main(
            [
                str(scene),
                "--sgw-root",
                str(sgw_root),
                "--iteration",
                "12000",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "extract_mesh.py" in output
    assert "semantic_gaussian_wrapping_iteration_12000.ply" in output
