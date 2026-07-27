"""Filesystem helpers kept intentionally small and deterministic."""

from __future__ import annotations

from pathlib import Path


def mkdir_p(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def search_for_max_iteration(folder: str | Path) -> int:
    root = Path(folder)
    iterations = []
    for path in root.glob("iteration_*"):
        try:
            iterations.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directory found in {root}")
    return max(iterations)

