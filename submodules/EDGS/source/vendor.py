"""Stable import paths for EDGS' vendored research dependencies.

The upstream Gaussian Splatting directory contains a dash and therefore cannot
be imported as a regular Python package.  Keeping the path bootstrap in one
place avoids cwd-dependent imports and, importantly, never exposes the PGSR
repository root (whose top-level module names collide with EDGS dependencies).
"""

from __future__ import annotations

import sys
from pathlib import Path

EDGS_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_SPLATTING_ROOT = EDGS_ROOT / "submodules" / "gaussian-splatting"
ROMA_ROOT = EDGS_ROOT / "submodules" / "RoMa"
MVROMA_ROOT = EDGS_ROOT / "submodules" / "MV-RoMa"
UFM_ROOT = EDGS_ROOT / "submodules" / "UFM"
UNICEPTION_ROOT = UFM_ROOT / "UniCeption"
DINOV2_ROOT = EDGS_ROOT / "submodules" / "DINOv2"


def _prepend(path: Path) -> None:
    resolved = str(path.resolve())
    # Reposition an existing entry too: callers may have inherited a
    # lower-priority PYTHONPATH entry from a notebook or launcher.
    while resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)


def bootstrap_gaussian_splatting() -> None:
    """Make EDGS' pinned Gaussian Splatting checkout importable."""

    _prepend(GAUSSIAN_SPLATTING_ROOT)


def bootstrap_roma() -> None:
    """Make EDGS' pinned RoMa checkout importable without a pip install."""

    _prepend(ROMA_ROOT)


def bootstrap_mvroma(root: Path | str | None = None) -> None:
    """Make the pinned MV-RoMa inference stack importable.

    MV-RoMa imports its model as ``src.*`` while its prematcher imports the
    complete top-level ``uniflowmatch`` and ``uniception`` packages.  Keep all
    three paths explicit so the paintmesh environment can use the vendored
    sources without changing its existing PyTorch/CUDA packages.
    """

    mvroma_root = Path(root).expanduser() if root is not None else MVROMA_ROOT
    _prepend(UNICEPTION_ROOT)
    _prepend(UFM_ROOT)
    _prepend(mvroma_root)
