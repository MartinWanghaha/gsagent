from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from training.engine import JsonlLogger, SemanticGaussianTrainer


def _record(iteration: int, loss: float = 0.0) -> str:
    return json.dumps({"iteration": iteration, "loss": loss}, sort_keys=True) + "\n"


def test_rewind_archives_abandoned_tail_and_never_overwrites_archive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.jsonl"
    retained = [_record(11_990), _record(12_000)]
    abandoned = [_record(12_010), _record(13_290)]
    path.write_text("".join([*retained, *abandoned]), encoding="utf8")
    logger = JsonlLogger(path)

    first_archive = logger.rewind(12_000)

    assert first_archive == tmp_path / "training.abandoned_after_12000.jsonl"
    assert path.read_text(encoding="utf8") == "".join(retained)
    assert first_archive.read_text(encoding="utf8") == "".join(abandoned)

    replacement_tail = _record(12_010, loss=1.0)
    with path.open("a", encoding="utf8") as stream:
        stream.write(replacement_tail)
    second_archive = logger.rewind(12_000)

    assert second_archive == tmp_path / "training.abandoned_after_12000_001.jsonl"
    assert first_archive.read_text(encoding="utf8") == "".join(abandoned)
    assert second_archive.read_text(encoding="utf8") == replacement_tail
    assert path.read_text(encoding="utf8") == "".join(retained)
    assert not list(tmp_path.glob(".*.tmp"))


def test_rewind_is_noop_without_a_log_or_future_records(tmp_path: Path) -> None:
    assert JsonlLogger(None).rewind(12_000) is None
    assert JsonlLogger(tmp_path / "missing.jsonl").rewind(12_000) is None

    path = tmp_path / "training.jsonl"
    original = _record(11_990) + _record(12_000)
    path.write_text(original, encoding="utf8")

    assert JsonlLogger(path).rewind(12_000) is None
    assert path.read_text(encoding="utf8") == original
    assert list(tmp_path.glob("training.abandoned_after_*.jsonl")) == []


def test_rewind_preserves_a_truncated_crash_tail_in_archive(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    retained = _record(12_000)
    truncated = '{"iteration": 12010, "loss":'
    path.write_text(retained + truncated, encoding="utf8")

    archive = JsonlLogger(path).rewind(12_000)

    assert archive is not None
    assert path.read_text(encoding="utf8") == retained
    assert archive.read_text(encoding="utf8") == truncated


class _RecordingLogger:
    def __init__(self) -> None:
        self.start_iterations: list[int] = []

    def rewind(self, start_iteration: int) -> None:
        self.start_iterations.append(start_iteration)


def _empty_trainer(total: int, logger: _RecordingLogger) -> SemanticGaussianTrainer:
    trainer = object.__new__(SemanticGaussianTrainer)
    trainer.scene = SimpleNamespace(
        getTrainCameras=lambda: [SimpleNamespace(uid=0)]
    )
    trainer.config = {
        "optimization": {"iterations": total},
        "density": {},
        "logging": {"log_interval": 10},
    }
    trainer.logger = logger
    trainer.gaussians = SimpleNamespace(get_xyz=torch.empty((1, 3)))
    trainer._camera_stack_ids = []
    trainer.last_metrics = {}
    return trainer


def test_train_rewinds_log_exactly_once_only_for_resume() -> None:
    resumed_logger = _RecordingLogger()
    result = _empty_trainer(12_000, resumed_logger).train(start_iteration=12_000)
    assert result.iteration == 12_000
    assert resumed_logger.start_iterations == [12_000]

    fresh_logger = _RecordingLogger()
    _empty_trainer(0, fresh_logger).train(start_iteration=0)
    assert fresh_logger.start_iterations == []
