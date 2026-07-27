from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


GSAGENT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = GSAGENT_ROOT / "scripts" / "gaga" / "associate_gaga_masks_mipnerf360.py"
SPEC = importlib.util.spec_from_file_location("gaga_mipnerf360_association", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_memory_efficient_projector_defines_visualize_when_disabled(
    tmp_path, monkeypatch
) -> None:
    gaga_root = tmp_path / "Gaga"
    (gaga_root / "mask").mkdir(parents=True)
    (gaga_root / "scene").mkdir()
    (gaga_root / "mask" / "projector.py").touch()
    (gaga_root / "scene" / "gaussian_model.py").touch()

    mask_package = ModuleType("mask")
    mask_package.__path__ = []
    projector_module = ModuleType("mask.projector")

    class LegacyGaussianProjector:
        def __init__(self, dataset, pipeline, iteration, params, device) -> None:
            del dataset, pipeline, iteration, params
            self.device = device
            self.image_width = 8
            self.image_height = 6
            self.patch_mask = object()
            self.flatten_patch_mask = object()
            # Reproduce Gaga's legacy behavior: no `visualize` attribute when
            # visualization is disabled.

    projector_module.GaussianProjector = LegacyGaussianProjector
    monkeypatch.setitem(sys.modules, "mask", mask_package)
    monkeypatch.setitem(sys.modules, "mask.projector", projector_module)

    projector_class = launcher.load_projector_class(gaga_root)
    projector = projector_class(
        SimpleNamespace(),
        SimpleNamespace(),
        iteration=1,
        params={"num_patch": 2, "visualize": False},
        device="cpu",
    )

    assert projector.visualize is False
    assert projector.num_patches == 2
    assert projector.patch_width == 4
    assert projector.patch_height == 3
