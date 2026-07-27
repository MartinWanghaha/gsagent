"""Reproducible experiment resume and random-state serialization."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from utils.config_utils import apply_overrides, load_config, validate_config


RESUME_OVERRIDE_KEYS = {
    "optimization.iterations",
    # Pure execution-memory policy: changing it does not alter logits, labels,
    # losses, topology budgets, or checkpointed model state.
    "semantic.region_decode_chunk_size",
    "surface.support_routing_query_chunk",
    "surface.scipy_workers",
    "surface.mesh_feedback_scipy_workers",
}
RESUME_OVERRIDE_PREFIXES = ("logging.",)
TRAINING_CHECKPOINT_VERSION = 3


def validate_training_checkpoint_header(state: Any) -> int:
    """Validate the exact native checkpoint header and return its iteration."""

    if not isinstance(state, dict):
        raise ValueError("training checkpoint must be a mapping")
    version = state.get("version")
    if type(version) is not int:
        raise ValueError("training checkpoint has an invalid schema version")
    if version > TRAINING_CHECKPOINT_VERSION:
        raise ValueError("training checkpoint was produced by a newer schema")
    if version != TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint does not contain the region-conditioned training schema; "
            "start a fresh run"
        )
    iteration = state.get("iteration")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("training checkpoint has an invalid iteration")
    return iteration


def _load_resume_checkpoint(checkpoint_path: str | Path) -> Any:
    """Load a CPU resume state without eagerly faulting every tensor page.

    Native training checkpoints contain the model, Adam moments and density
    windows.  Keeping their storages file-backed until restore avoids adding
    the entire checkpoint to the already high scene-construction RSS.  Older
    PyTorch releases do not accept the ``mmap`` keyword; retry only that API
    compatibility failure with the historical eager loader.
    """

    load_options = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(checkpoint_path, mmap=True, **load_options)
    except TypeError:
        return torch.load(checkpoint_path, **load_options)


def atomic_torch_save(value: Any, destination: str | Path) -> None:
    """Durably replace one checkpoint only after serialization succeeds.

    ``torch.save`` writes a multi-part archive and an interrupted write can
    otherwise leave a correctly named but unusable resume checkpoint.  The
    temporary file lives beside the destination so ``os.replace`` is an
    atomic same-filesystem operation.  Any previous checkpoint is preserved
    when serialization fails.
    """

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        # Persist the directory entry where the platform supports directory
        # fsync.  The checkpoint is already atomically visible if this fails.
        try:
            directory_descriptor = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_training_configuration(
    *,
    default_config: str | Path,
    config_path: str | Path | None,
    overrides: list[str] | None,
    checkpoint_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve a fresh or resumed experiment without configuration drift.

    A native checkpoint owns the resolved optimization definition. Resume may
    extend the total iteration count, change logging cadence, or adjust the
    exact semantic-routing chunk size, but objective, phase, density, model and
    data changes fail fast.
    """

    if checkpoint_path is None:
        config = load_config(config_path or default_config, overrides)
        validate_config(config)
        return config, None
    if config_path is not None:
        raise ValueError(
            "--config cannot be combined with --checkpoint: resume uses the "
            "resolved config embedded in the checkpoint; use the allowed --set options"
        )
    state = _load_resume_checkpoint(checkpoint_path)
    saved_iteration = validate_training_checkpoint_header(state)
    if not isinstance(state.get("config"), dict):
        raise ValueError("checkpoint does not contain a resolved training config")
    config = apply_overrides(
        copy.deepcopy(state["config"]),
        overrides,
        allowed_keys=RESUME_OVERRIDE_KEYS,
        allowed_prefixes=RESUME_OVERRIDE_PREFIXES,
    )
    validate_config(config)
    if int(config["optimization"]["iterations"]) < saved_iteration:
        raise ValueError(
            "optimization.iterations cannot precede checkpoint iteration "
            f"{saved_iteration}"
        )
    return config, state


def validate_resume_source(
    config: dict[str, Any],
    source_path: str | Path,
    *,
    allow_relocation: bool = False,
) -> None:
    if allow_relocation:
        return
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("checkpoint config has no runtime mapping")
    saved = runtime.get("source_path")
    if not isinstance(saved, str) or not saved:
        raise ValueError("checkpoint config has no runtime.source_path")
    if Path(saved).resolve() != Path(source_path).resolve():
        raise ValueError(
            "resume source differs from the checkpoint dataset; pass "
            "--allow-source-relocation only when the dataset is an exact copy"
        )


def capture_rng_state() -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": keys.astype(np.uint32).tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    cuda_states = state.get("cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


__all__ = [
    "RESUME_OVERRIDE_KEYS",
    "atomic_torch_save",
    "capture_rng_state",
    "resolve_training_configuration",
    "restore_rng_state",
    "validate_training_checkpoint_header",
    "validate_resume_source",
]
