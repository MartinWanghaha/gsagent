"""Strictly local DINOv2 loading for UniCeption.

UniCeption currently hard-codes ``torch.hub.load("facebookresearch/dinov2",
...)``.  Replacing that vendored project with a fork would make the nested
submodule difficult to maintain, so EDGS redirects only that exact call while
UFM is being constructed.  Source and weights are both explicit local files;
there is no remote or cache fallback.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch

_REMOTE_DINOV2_REPOSITORY = "facebookresearch/dinov2"
_SUPPORTED_MODEL = "dinov2_vitl14"
_TORCH_HUB_REDIRECT_LOCK = threading.RLock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for wrapper in ("state_dict", "model", "model_state_dict"):
            candidate = payload.get(wrapper)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise TypeError("DINOv2 checkpoint must contain a state dictionary")
    return {
        (str(name)[7:] if str(name).startswith("module.") else str(name)): value
        for name, value in payload.items()
    }


def _torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch releases predating ``weights_only``.
        return torch.load(path, map_location="cpu")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class LocalDINOv2:
    """Identity and loader for the vendored ViT-L/14 backbone."""

    root: Path
    checkpoint: Path
    checkpoint_sha256: str | None = None

    def validate(self) -> None:
        root = self.root.resolve()
        checkpoint = self.checkpoint.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DINOv2 source directory does not exist: {root}")
        if (
            not (root / "hubconf.py").is_file()
            or not (root / "dinov2" / "hub" / "backbones.py").is_file()
        ):
            raise FileNotFoundError(f"invalid vendored DINOv2 checkout: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"DINOv2 checkpoint does not exist: {checkpoint}. "
                "Set init_wC.mvroma.dinov2_checkpoint to a local ViT-L/14 checkpoint."
            )
        if self.checkpoint_sha256 is not None:
            actual = sha256_file(checkpoint)
            if actual.lower() != self.checkpoint_sha256.lower():
                raise ValueError(
                    "DINOv2 checkpoint SHA-256 mismatch: "
                    f"expected {self.checkpoint_sha256}, got {actual}"
                )

    def load(
        self,
        original_hub_load: Callable[..., Any],
        model_name: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Build from vendored source, then strictly load the local weights."""

        if model_name != _SUPPORTED_MODEL:
            raise ValueError(
                "the configured local DINOv2 checkpoint supports only "
                f"{_SUPPORTED_MODEL!r}; UniCeption requested {model_name!r}"
            )

        # These options belong to GitHub-backed torch.hub resolution and must
        # never leak into the local entry point.  ``pretrained=False`` is
        # mandatory: DINOv2's pretrained=True path downloads from Meta's CDN.
        for name in ("source", "force_reload", "trust_repo", "skip_validation"):
            kwargs.pop(name, None)
        kwargs.pop("pretrained", None)

        print(
            f"Loading {_SUPPORTED_MODEL} from vendored source {self.root.resolve()} "
            f"and local checkpoint {self.checkpoint.resolve()}"
        )
        model = original_hub_load(
            str(self.root.resolve()),
            model_name,
            *model_args,
            source="local",
            pretrained=False,
            **kwargs,
        )
        model_source = Path(inspect.getfile(model.__class__)).resolve()
        if not _is_within(model_source, self.root.resolve()):
            raise RuntimeError(
                "DINOv2 Python module was not imported from the configured vendored "
                f"checkout: module={model_source}, root={self.root.resolve()}"
            )

        payload = _torch_load_weights(self.checkpoint.resolve())
        model.load_state_dict(_checkpoint_state(payload), strict=True)
        return model

    @contextmanager
    def redirect_uniception_hub_call(self) -> Iterator[None]:
        """Redirect only UniCeption's DINOv2 GitHub Hub request.

        ``torch.hub.load`` is process-global, therefore the short construction
        window is serialized and restored in ``finally``.  Other repositories
        continue to use the original function unchanged.
        """

        with _TORCH_HUB_REDIRECT_LOCK:
            original_hub_load = torch.hub.load

            def local_hub_load(
                repo_or_dir: str | Path,
                model_name: str,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                repository = str(repo_or_dir)
                if repository == _REMOTE_DINOV2_REPOSITORY or repository.startswith(
                    f"{_REMOTE_DINOV2_REPOSITORY}:"
                ):
                    return self.load(original_hub_load, model_name, *args, **kwargs)
                return original_hub_load(repo_or_dir, model_name, *args, **kwargs)

            torch.hub.load = local_hub_load
            try:
                yield
            finally:
                torch.hub.load = original_hub_load
