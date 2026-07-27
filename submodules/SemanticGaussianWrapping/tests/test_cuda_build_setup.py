from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import setuptools
import torch

import install as installer


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "submodules" / "diff-semantic-gaussian-rasterization"


def _capture_extension_setup(monkeypatch, architecture: str | None) -> dict[str, object]:
    captured: dict[str, object] = {}
    if architecture is None:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", architecture)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(EXTENSION / "setup.py"), run_name="__main__")
    return captured


def test_extension_setup_is_headless_and_pep517_safe(monkeypatch) -> None:
    captured = _capture_extension_setup(monkeypatch, None)

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.5;8.0;8.6+PTX"
    extension = captured["ext_modules"][0]
    assert extension.sources == ["ext.cpp", "cuda_rasterizer/rasterize.cu"]
    assert all(not Path(source).is_absolute() for source in extension.sources)


def test_explicit_architecture_is_preserved(monkeypatch) -> None:
    _capture_extension_setup(monkeypatch, "8.9")
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_installer_forwards_cuda_architecture(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    monkeypatch.setattr(installer, "run", lambda command, env=None: calls.append((command, env)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install.py",
            "--skip-dependencies",
            "--cuda-arch-list",
            "8.6",
            "--no-editable",
        ],
    )

    installer.main()

    assert len(calls) == 1
    command, environment = calls[0]
    assert "diff-semantic-gaussian-rasterization" in " ".join(command)
    assert environment is not None
    assert environment["TORCH_CUDA_ARCH_LIST"] == "8.6"
