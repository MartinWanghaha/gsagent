from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
import torch

from metrics import (
    _write_json_preserving,
    evaluate_directory,
    resolve_metric_device,
    result_output_path,
    select_methods,
)


def test_metrics_can_select_several_explicit_or_all_iterations(tmp_path) -> None:
    for iteration in (7000, 12000, 24000):
        (tmp_path / "test" / f"ours_{iteration}").mkdir(parents=True)
    (tmp_path / "test" / "ours_invalid").mkdir()

    selected = select_methods(tmp_path, "test", [7000, 24000])
    assert [path.name for path in selected] == ["ours_7000", "ours_24000"]
    assert select_methods(tmp_path, "test")[0].name == "ours_24000"
    assert [path.name for path in select_methods(tmp_path, "test", all_iterations=True)] == [
        "ours_7000",
        "ours_12000",
        "ours_24000",
    ]
    with pytest.raises(FileNotFoundError, match="12001"):
        select_methods(tmp_path, "test", [12001])


def test_metric_outputs_are_iteration_scoped_and_never_replace_different_results(tmp_path) -> None:
    path = result_output_path(tmp_path, "test", 7000)
    first = _write_json_preserving(path, {"psnr": 20.0})
    same = _write_json_preserving(path, {"psnr": 20.0})
    second = _write_json_preserving(path, {"psnr": 21.0})

    assert first == same == path
    assert second.name == "ours_7000_001.json"
    assert json.loads(path.read_text(encoding="utf8")) == {"psnr": 20.0}
    assert json.loads(second.read_text(encoding="utf8")) == {"psnr": 21.0}


def test_metrics_ignore_stale_files_outside_latest_render_selection(tmp_path) -> None:
    method = tmp_path / "test" / "ours_7"
    (method / "renders").mkdir(parents=True)
    (method / "gt").mkdir()
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    Image.fromarray(black).save(method / "renders" / "00000.png")
    Image.fromarray(white).save(method / "gt" / "00000.png")
    Image.fromarray(white).save(method / "renders" / "00001.png")
    Image.fromarray(white).save(method / "gt" / "00001.png")
    (method / "metadata.json").write_text(
        json.dumps({"view_indices": [1], "num_semantic_classes": 0}),
        encoding="utf8",
    )
    result = evaluate_directory(method, torch.device("cpu"), compute_lpips=False)
    assert result["views"] == 1
    assert result["l1"] == 0.0


def test_metric_cuda_index_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_metric_device("cuda:3") == torch.device("cpu")
