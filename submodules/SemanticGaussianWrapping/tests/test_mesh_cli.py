from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import extract_mesh
from mesh import TriangleMesh, load_mesh


def _arguments(*values: str):
    parser = extract_mesh.build_parser()
    args = parser.parse_args(["-m", "trained_scene", "--output", "surface.ply", *values])
    extract_mesh._validate_args(parser, args)
    return args


@pytest.mark.parametrize(
    "arguments",
    [
        ["--method", "marching_cubes"],
        ["--resolution", "256"],
        ["--scalar", "sdf"],
        ["--level", "0"],
        ["--region-id", "3"],
        ["--all-regions"],
        ["--field-factory", "module:factory"],
        ["--bounds", "-1", "-1", "-1", "1", "1", "1"],
        ["--query-chunk-size", "1024"],
        ["--scipy-workers", "4"],
        ["--max-pivots", "120000"],
    ],
)
def test_parser_rejects_removed_extraction_paths(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        _arguments(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--iteration", "0"],
        ["--iteration", "-2"],
        ["--output", "surface.obj"],
        ["--max-gaussians", "3"],
        ["--max-chart-gaussians", "3"],
        ["--view-stride", "0"],
        ["--camera-scale", "0"],
        ["--camera-scale", "1.01"],
        ["--target-faces", "0"],
        ["--metric-threshold", "0"],
        ["--metric-threshold", "nan"],
        ["--metric-samples", "0"],
        ["--metric-seed", "-1"],
        ["--metrics-json", "metrics.json"],
    ],
)
def test_parser_rejects_invalid_rcgw_or_metric_options(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        _arguments(*arguments)


def test_checkpoint_rcgw_config_is_authoritative_and_cli_wins() -> None:
    context = SimpleNamespace(
        experiment_config={
            "mesh_export": {
                "max_gaussians": 120_000,
                "max_chart_gaussians": 8_000,
                "view_stride": 3,
                "camera_scale": 0.5,
                "target_faces": 250_000,
                "candidate_views": 3,
                "binary_steps": 12,
            }
        }
    )
    args = _arguments(
        "--max-gaussians",
        "150000",
        "--max-chart-gaussians",
        "10000",
        "--camera-scale",
        "0.75",
    )

    config = extract_mesh._resolved_config(context, args)

    assert isinstance(config, extract_mesh.RegionGaussianWrappingConfig)
    assert config.max_gaussians == 150_000
    assert config.max_chart_gaussians == 10_000
    assert config.view_stride == 3
    assert config.camera_scale == pytest.approx(0.75)
    assert config.target_faces == 250_000
    assert config.candidate_views == 3
    assert config.binary_steps == 12


def _context(tmp_path: Path):
    model_path = tmp_path / "trained_scene"
    model_path.mkdir()
    checkpoint = model_path / "chkpnt30000.pth"
    checkpoint.touch()
    return SimpleNamespace(
        model_path=model_path,
        checkpoint_path=checkpoint,
        iteration=30_000,
        cameras=(object(), object(), object()),
        experiment_config={
            "mesh_export": {
                "max_gaussians": 120,
                "max_chart_gaussians": 40,
                "view_stride": 1,
                "camera_scale": 1.0,
            }
        },
    )


def _triangle_mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        normals=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        semantic_id=np.asarray([2, 2, 2], dtype=np.int64),
        semantic=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        uncertainty=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        face_region_id=np.asarray([2], dtype=np.int64),
        metadata={"charts": 3, "shared_roots": 12},
    )


def test_main_uses_one_rcgw_pipeline_and_publishes_binary_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    observed: dict[str, object] = {}

    def load_context(model_path, *, iteration, device):
        observed["load"] = (model_path, iteration, device)
        return context

    class Extractor:
        def __init__(self, loaded_context, *, config, progress_callback):
            observed["context"] = loaded_context
            observed["config"] = config
            observed["progress_callback"] = progress_callback

        def extract(self):
            observed["extract_calls"] = int(observed.get("extract_calls", 0)) + 1
            return _triangle_mesh()

    monkeypatch.setattr(
        extract_mesh.MeshExtractionContext,
        "load",
        staticmethod(load_context),
    )
    monkeypatch.setattr(
        extract_mesh,
        "RegionConditionedGaussianWrappingExtractor",
        Extractor,
    )
    output = tmp_path / "mesh" / "surface.ply"

    assert (
        extract_mesh.main(
            [
                "-m",
                str(context.model_path),
                "--iteration",
                "30000",
                "--output",
                str(output),
                "--device",
                "cpu",
                "--max-gaussians",
                "150",
                "--max-chart-gaussians",
                "50",
            ]
        )
        == 0
    )

    assert observed["load"] == (str(context.model_path), 30_000, "cpu")
    assert observed["context"] is context
    assert observed["config"].max_gaussians == 150
    assert observed["config"].max_chart_gaussians == 50
    assert observed["extract_calls"] == 1
    assert callable(observed["progress_callback"])
    assert output.read_bytes().startswith(b"ply\nformat binary_little_endian")
    restored = load_mesh(output)
    np.testing.assert_array_equal(restored.semantic_id, [2, 2, 2])

    manifest_path = output.with_suffix(".ply.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["algorithm"] == "region_conditioned_gaussian_wrapping"
    assert payload["iteration"] == 30_000
    assert payload["train_cameras"] == 3
    assert payload["vertices"] == 3 and payload["faces"] == 1
    assert payload["metadata"]["charts"] == 3
    assert payload["extraction_config"]["max_gaussians"] == 150
    assert payload["extraction_config"]["max_chart_gaussians"] == 50


def test_atomic_mesh_publication_keeps_previous_output_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "surface.ply"
    output.write_bytes(b"previous")

    def fail_after_partial_write(_mesh, temporary):
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("simulated exporter failure")

    monkeypatch.setattr(extract_mesh, "export_mesh", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated exporter failure"):
        extract_mesh._atomic_export(_triangle_mesh(), output)

    assert output.read_bytes() == b"previous"
    assert list(tmp_path.glob(".surface.ply.*")) == []


def test_optional_reference_metrics_are_recorded_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    reference = tmp_path / "reference.xyz"
    reference.write_text("0 0 0\n", encoding="utf-8")
    output = tmp_path / "surface.ply"
    metrics_output = tmp_path / "metrics.json"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        extract_mesh.MeshExtractionContext,
        "load",
        staticmethod(lambda *args, **kwargs: context),
    )

    class Extractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract(self):
            return _triangle_mesh()

    class Metrics:
        def as_dict(self):
            return {"accuracy": 0.01, "completeness": 0.02, "fscore": 0.9}

    def load_reference(path):
        observed["reference"] = path
        return np.zeros((1, 3), dtype=np.float32)

    def compute(mesh, target, *, threshold, sample_count, seed):
        observed["metrics"] = (mesh, target, threshold, sample_count, seed)
        return Metrics()

    monkeypatch.setattr(
        extract_mesh,
        "RegionConditionedGaussianWrappingExtractor",
        Extractor,
    )
    monkeypatch.setattr(extract_mesh, "_reference_geometry", load_reference)
    monkeypatch.setattr(extract_mesh, "compute_mesh_metrics", compute)

    assert (
        extract_mesh.main(
            [
                "-m",
                str(context.model_path),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--reference",
                str(reference),
                "--metric-threshold",
                "0.02",
                "--metric-samples",
                "4096",
                "--metric-seed",
                "7",
                "--metrics-json",
                str(metrics_output),
            ]
        )
        == 0
    )

    assert observed["reference"] == str(reference)
    _, target, threshold, sample_count, seed = observed["metrics"]
    np.testing.assert_array_equal(target, np.zeros((1, 3), dtype=np.float32))
    assert (threshold, sample_count, seed) == (0.02, 4096, 7)
    metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
    assert metrics["fscore"] == pytest.approx(0.9)
    manifest = json.loads(output.with_suffix(".ply.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["metrics"] == metrics
