"""Same-process, two-phase MV-RoMa runtime for the ``paintmesh`` environment.

The prematcher is run for all camera groups first and only its small track-token
input is retained in CPU memory. It is then released before the much larger
MV-RoMa model is constructed. Dense flow/certainty is consumed group by group;
the caller may merge it into a lossless same-run pair store, but no reusable
sparse correspondence cache is written.
"""

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from source.vendor import DINOV2_ROOT, EDGS_ROOT, bootstrap_mvroma

from .config import as_bool, config_value
from .contracts import (
    DenseMultiViewCorrespondence,
    ImageSamplingGeometry,
    normalized_image_grid,
)
from .local_dinov2 import LocalDINOv2, sha256_file
from .planning import CameraGroup

_MISSING = object()


def _get(config: Any, name: str, default: Any = _MISSING) -> Any:
    """Backward-compatible alias used by the existing adapter/tests."""

    if default is _MISSING:
        return config_value(config, name)
    return config_value(config, name, default)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"none", "null"} else text


def _resolution(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        values = value.lower().replace("x", ",").split(",")
    else:
        values = list(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain [height,width]")
    result = int(values[0]), int(values[1])
    if min(result) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return result


@dataclass(frozen=True)
class MVRoMaSettings:
    root: Path
    checkpoint: Path
    dinov2_root: Path
    dinov2_checkpoint: Path
    checkpoint_sha256: str | None = None
    source_revision: str | None = None
    dinov2_checkpoint_sha256: str | None = None
    dinov2_source_revision: str | None = None
    prematcher: str = "ufm"
    prematcher_model_id: str = "infinity1096/UFM-Refine"
    prematcher_revision: str | None = None
    prematcher_local_files_only: bool = True
    prematch_resolution: tuple[int, int] = (420, 560)
    coarse_resolution: tuple[int, int] = (560, 560)
    target_resolution: tuple[int, int] = (560, 840)
    num_clusters: int = 512
    track_cluster_mode: str = "global"
    cluster_downsample_stride: int = 4
    cluster_kmeans_iters: int = 12
    cluster_kmeans_seed: int = 0
    upsample_preds: bool = True
    apply_square: bool = False
    prematch_batched: bool = False
    covisibility_threshold: float = 0.3
    strict_checkpoint: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "MVRoMaSettings":
        root_value = _optional_text(_get(config, "root", None))
        root = (
            Path(root_value).expanduser()
            if root_value is not None
            else EDGS_ROOT / "submodules" / "MV-RoMa"
        )
        checkpoint_value = _optional_text(_get(config, "checkpoint", None))
        if checkpoint_value is None:
            raise ValueError("init_wC.mvroma.checkpoint must point to MV-RoMa weights")
        dinov2_root_value = _optional_text(_get(config, "dinov2_root", None))
        dinov2_root = (
            Path(dinov2_root_value).expanduser()
            if dinov2_root_value is not None
            else DINOV2_ROOT
        )
        dinov2_checkpoint_value = _optional_text(
            _get(config, "dinov2_checkpoint", None)
        )
        if dinov2_checkpoint_value is None:
            raise ValueError(
                "init_wC.mvroma.dinov2_checkpoint must point to local "
                "DINOv2 ViT-L/14 weights"
            )
        settings = cls(
            root=root,
            checkpoint=Path(checkpoint_value).expanduser(),
            dinov2_root=dinov2_root,
            dinov2_checkpoint=Path(dinov2_checkpoint_value).expanduser(),
            checkpoint_sha256=_optional_text(_get(config, "checkpoint_sha256", None)),
            source_revision=_optional_text(_get(config, "source_revision", None)),
            dinov2_checkpoint_sha256=_optional_text(
                _get(config, "dinov2_checkpoint_sha256", None)
            ),
            dinov2_source_revision=_optional_text(
                _get(config, "dinov2_source_revision", None)
            ),
            prematcher=str(_get(config, "prematcher", "ufm")).lower(),
            prematcher_model_id=str(
                _get(config, "prematcher_model_id", "infinity1096/UFM-Refine")
            ),
            prematcher_revision=_optional_text(
                _get(config, "prematcher_revision", None)
            ),
            prematcher_local_files_only=as_bool(
                _get(config, "prematcher_local_files_only", True),
                name="prematcher_local_files_only",
            ),
            prematch_resolution=_resolution(
                _get(config, "prematch_resolution", (420, 560)),
                "prematch_resolution",
            ),
            coarse_resolution=_resolution(
                _get(config, "coarse_resolution", (560, 560)), "coarse_resolution"
            ),
            target_resolution=_resolution(
                _get(config, "target_resolution", (560, 840)), "target_resolution"
            ),
            num_clusters=int(_get(config, "num_clusters", 512)),
            track_cluster_mode=str(
                _get(config, "track_cluster_mode", "global")
            ).lower(),
            cluster_downsample_stride=int(
                _get(config, "cluster_downsample_stride", 4)
            ),
            cluster_kmeans_iters=int(_get(config, "cluster_kmeans_iters", 12)),
            cluster_kmeans_seed=int(_get(config, "cluster_kmeans_seed", 0)),
            upsample_preds=as_bool(
                _get(config, "upsample_preds", True), name="upsample_preds"
            ),
            apply_square=as_bool(
                _get(config, "apply_square", False), name="apply_square"
            ),
            prematch_batched=as_bool(
                _get(config, "prematch_batched", False), name="prematch_batched"
            ),
            covisibility_threshold=float(_get(config, "covisibility_threshold", 0.3)),
            strict_checkpoint=as_bool(
                _get(config, "strict_checkpoint", False),
                name="strict_checkpoint",
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.prematcher != "ufm":
            raise ValueError(
                "the audited MV-RoMa implementation currently supports only "
                "prematcher='ufm'"
            )
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if self.track_cluster_mode not in {
            "global",
            "per-group",
            "visibility_partition",
        }:
            raise ValueError(
                "track_cluster_mode must be global or visibility_partition"
            )
        if self.cluster_downsample_stride <= 0 or self.cluster_kmeans_iters <= 0:
            raise ValueError("cluster stride and k-means iterations must be positive")
        if not 0.0 <= self.covisibility_threshold <= 1.0:
            raise ValueError("covisibility_threshold must be in [0,1]")
        patch_size = 14
        invalid_resolutions = [
            name
            for name, resolution in (
                ("coarse_resolution", self.coarse_resolution),
                ("target_resolution", self.target_resolution),
            )
            if any(dimension % patch_size for dimension in resolution)
        ]
        if invalid_resolutions:
            raise ValueError(
                f"{', '.join(invalid_resolutions)} must be divisible by "
                f"MV-RoMa patch size {patch_size}"
            )


@dataclass(frozen=True)
class InferenceGroup:
    cameras: CameraGroup
    source_path: Path
    target_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if len(self.target_paths) != len(self.cameras.target_indices):
            raise ValueError("target paths and target camera indices differ in length")
        if not self.target_paths:
            raise ValueError("MV-RoMa requires at least one target")

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return (self.source_path, *self.target_paths)


@dataclass(frozen=True)
class PreparedInferenceGroup:
    group: InferenceGroup
    paired_tracks: torch.Tensor

    def __post_init__(self) -> None:
        targets = len(self.group.target_paths)
        if self.paired_tracks.ndim != 3 or self.paired_tracks.shape != (
            targets,
            self.paired_tracks.shape[1],
            4,
        ):
            raise ValueError("paired_tracks must have shape [T,M,4]")
        if self.paired_tracks.device.type != "cpu":
            raise ValueError("prepared track tokens must reside in CPU memory")


@dataclass(frozen=True)
class _Padding:
    original_width: int
    original_height: int
    square_size: int
    left: int
    top: int
    right: int
    bottom: int


PrematcherFactory = Callable[[str, torch.device], Any]
TrackProvider = Callable[
    [Any, InferenceGroup, MVRoMaSettings, torch.device], torch.Tensor
]
ModelFactory = Callable[[MVRoMaSettings, torch.device], torch.nn.Module]
DenseRunner = Callable[
    [torch.nn.Module, PreparedInferenceGroup, MVRoMaSettings, torch.device],
    DenseMultiViewCorrespondence,
]


def _validate_sources(settings: MVRoMaSettings) -> None:
    if not settings.root.is_dir():
        raise FileNotFoundError(
            f"MV-RoMa source directory does not exist: {settings.root}"
        )
    if not (settings.root / "src" / "build_model.py").is_file():
        raise FileNotFoundError(f"invalid MV-RoMa checkout: {settings.root}")
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(
            f"MV-RoMa checkpoint does not exist: {settings.checkpoint}"
        )
    if settings.checkpoint_sha256 is not None:
        actual = sha256_file(settings.checkpoint)
        if actual.lower() != settings.checkpoint_sha256.lower():
            raise ValueError(
                "MV-RoMa checkpoint SHA-256 mismatch: "
                f"expected {settings.checkpoint_sha256}, got {actual}"
            )
    LocalDINOv2(
        root=settings.dinov2_root,
        checkpoint=settings.dinov2_checkpoint,
        checkpoint_sha256=settings.dinov2_checkpoint_sha256,
    ).validate()
    # ``source_revision`` is a launcher provenance identity.  It may describe
    # the composite MV-RoMa/UFM/UniCeption tree (including dirty state), not
    # merely MV-RoMa's HEAD.  The run_seg reuse contract validates that value;
    # comparing it to one checkout's ``git rev-parse HEAD`` here would reject a
    # valid composite identity.


def _default_prematcher_factory(
    name: str,
    device: torch.device,
    *,
    settings: MVRoMaSettings,
) -> Any:
    try:
        local_dinov2 = LocalDINOv2(
            root=settings.dinov2_root,
            checkpoint=settings.dinov2_checkpoint,
            checkpoint_sha256=settings.dinov2_checkpoint_sha256,
        )
        with local_dinov2.redirect_uniception_hub_call():
            if name != "ufm":
                raise ValueError(f"unsupported MV-RoMa prematcher: {name!r}")
            module = importlib.import_module(
                "src.matchers.uniflowmatch.models.ufm"
            )
            model_class = module.UniFlowMatchClassificationRefinement
            model = model_class.from_pretrained(
                settings.prematcher_model_id,
                revision=settings.prematcher_revision,
                local_files_only=settings.prematcher_local_files_only,
            )
            return [None, model.eval().to(device)]
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "MV-RoMa's UFM prematcher could not be imported. Ensure the vendored "
            "UFM and UFM/UniCeption checkouts are initialized."
        ) from error


def _default_track_provider(
    prematcher: Any,
    group: InferenceGroup,
    settings: MVRoMaSettings,
    device: torch.device,
) -> torch.Tensor:
    matcher_module = importlib.import_module("src.matchers.run_matcher_path")
    cluster_module = importlib.import_module("src.track_cluster")
    image_dict = {
        "query_img_path": str(group.source_path),
        "ref_img_paths": [str(path) for path in group.target_paths],
    }
    match_height, match_width = settings.prematch_resolution
    query, targets, covisibility = matcher_module.run_match_multi_path(
        image_dict,
        match_W=match_width,
        match_H=match_height,
        matcher_model=prematcher,
        matcher_name=settings.prematcher,
        device=str(device),
        batched=settings.prematch_batched,
    )
    query = query.clone()
    targets = targets.clone()
    coarse_height, coarse_width = settings.coarse_resolution
    query[..., 0] *= coarse_width / match_width
    query[..., 1] *= coarse_height / match_height
    targets[..., 0] *= coarse_width / match_width
    targets[..., 1] *= coarse_height / match_height
    return cluster_module.extract_tracks_match(
        query,
        targets,
        covisibility,
        N=settings.num_clusters,
        downsample_stride=settings.cluster_downsample_stride,
        covisibility_threshold=settings.covisibility_threshold,
        kmeans_iters=settings.cluster_kmeans_iters,
        kmeans_seed=settings.cluster_kmeans_seed,
        cluster_mode=(
            "per-group"
            if settings.track_cluster_mode == "visibility_partition"
            else settings.track_cluster_mode
        ),
        device=device,
    )


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for name in ("state_dict", "model", "model_state_dict"):
            candidate = payload.get(name)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise TypeError("MV-RoMa checkpoint must contain a state dictionary")
    state = {}
    for name, value in payload.items():
        key = name[7:] if str(name).startswith("module.") else str(name)
        state[key] = value
    return state


def _default_model_factory(
    settings: MVRoMaSettings,
    device: torch.device,
) -> torch.nn.Module:
    build_module = importlib.import_module("src.build_model")
    mvroma_module = importlib.import_module("src.mvroma")
    args = SimpleNamespace(
        use_dinov2=True,
        train_until_16x=False,
        train_refiner=False,
        train_all_model=False,
        num_cluster=settings.num_clusters,
    )
    model_config = mvroma_module.ModelConfig()
    model_config.num_cluster = settings.num_clusters
    model, _ = build_module.build_our_model(args, model_config, use_dinov2=True)
    payload = torch.load(settings.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(
        _checkpoint_state(payload), strict=settings.strict_checkpoint
    )
    if not settings.strict_checkpoint:
        missing = list(getattr(incompatible, "missing_keys", ()))
        unexpected = list(getattr(incompatible, "unexpected_keys", ()))
        if missing or unexpected:
            print(
                "MV-RoMa checkpoint loaded with strict=False: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    return model.eval().to(device)


def _padding_for(path: Path) -> _Padding:
    with Image.open(path) as image:
        width, height = image.size
    square = max(width, height)
    left = (square - width) // 2
    top = (square - height) // 2
    return _Padding(
        original_width=width,
        original_height=height,
        square_size=square,
        left=left,
        top=top,
        right=square - width - left,
        bottom=square - height - top,
    )


def _image_tensor(
    path: Path, size: tuple[int, int], padding: _Padding | None
) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if padding is not None:
            image = ImageOps.expand(
                image,
                border=(padding.left, padding.top, padding.right, padding.bottom),
                fill=0,
            )
        height, width = size
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((width, height), resample=resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _token_center_grid(
    batch: int,
    views: int,
    height: int,
    width: int,
    patch: int,
    device: torch.device,
) -> torch.Tensor:
    y = torch.arange(height // patch, device=device) * patch + patch / 2.0
    x = torch.arange(width // patch, device=device) * patch + patch / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).reshape(1, 1, -1, 2)
    return grid.repeat(batch, views, 1, 1)


def _adjust_tracks_for_square(
    paired_tracks: torch.Tensor,
    paddings: Sequence[_Padding],
    coarse_resolution: tuple[int, int],
) -> torch.Tensor:
    result = paired_tracks.clone()
    height, width = coarse_resolution

    def adjust(values: torch.Tensor, padding: _Padding) -> None:
        values[..., 0] = values[..., 0] * (
            padding.original_width / padding.square_size
        ) + padding.left * (width / padding.square_size)
        values[..., 1] = values[..., 1] * (
            padding.original_height / padding.square_size
        ) + padding.top * (height / padding.square_size)

    adjust(result[..., :2], paddings[0])
    for target_index, padding in enumerate(paddings[1:]):
        valid = result[target_index, :, 2:].mean(dim=-1) > -100.0
        if bool(valid.any().item()):
            adjusted = result[target_index, valid, 2:]
            adjust(adjusted, padding)
            result[target_index, valid, 2:] = adjusted
    return result


def _unpad_normalized(
    coordinates: torch.Tensor,
    padding: _Padding,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = coordinates.clone()
    x_square = (coordinates[..., 0] + 1.0) * (padding.square_size / 2.0) - 0.5
    y_square = (coordinates[..., 1] + 1.0) * (padding.square_size / 2.0) - 0.5
    x_original = x_square - padding.left
    y_original = y_square - padding.top
    result[..., 0] = (x_original + 0.5) * (2.0 / padding.original_width) - 1.0
    result[..., 1] = (y_original + 0.5) * (2.0 / padding.original_height) - 1.0
    valid = (
        torch.isfinite(result).all(dim=-1)
        & (result[..., 0].abs() <= 1.0)
        & (result[..., 1].abs() <= 1.0)
    )
    return result, valid


def _default_dense_runner(
    model: torch.nn.Module,
    prepared: PreparedInferenceGroup,
    settings: MVRoMaSettings,
    device: torch.device,
) -> DenseMultiViewCorrespondence:
    paths = prepared.group.image_paths
    image_shapes = tuple(_padding_for(path) for path in paths)
    paddings = image_shapes if settings.apply_square else None
    coarse = torch.stack(
        [
            _image_tensor(
                path,
                settings.coarse_resolution,
                None if paddings is None else paddings[i],
            )
            for i, path in enumerate(paths)
        ]
    )[None].to(device)
    high_resolution = None
    if settings.upsample_preds:
        high_resolution = torch.stack(
            [
                _image_tensor(
                    path,
                    settings.target_resolution,
                    None if paddings is None else paddings[i],
                )
                for i, path in enumerate(paths)
            ]
        )[None].to(device)

    paired_tracks = prepared.paired_tracks.to(device=device, dtype=torch.float32)
    if paddings is not None:
        paired_tracks = _adjust_tracks_for_square(
            paired_tracks, paddings, settings.coarse_resolution
        )
    paired_tracks = paired_tracks[None]
    patch_size = int(getattr(model, "patch_size", 14))
    coarse_height, coarse_width = settings.coarse_resolution
    feature_grid = _token_center_grid(
        1,
        len(paths),
        coarse_height,
        coarse_width,
        patch_size,
        device,
    )
    with torch.inference_mode():
        outputs = model.match(
            multi_view_frames=coarse,
            multi_view_frames_org=high_resolution,
            point_tracks=paired_tracks,
            feature_grid_coords=feature_grid,
            upsample_preds=settings.upsample_preds,
        )
    finest = min(outputs, key=lambda value: int(value))
    flow = outputs[finest]["flow"]
    logits = outputs[finest]["certainty"]
    if flow.ndim != 5 or logits.ndim != 5 or flow.shape[0] != 1 or logits.shape[0] != 1:
        raise ValueError("MV-RoMa must return flow/certainty shaped [1,T,C,H,W]")
    target_coordinates = flow[0]
    certainty_logits = logits[0]
    height, width = target_coordinates.shape[-2:]
    source_coordinates = normalized_image_grid(
        height,
        width,
        device=target_coordinates.device,
        dtype=target_coordinates.dtype,
    )

    if paddings is None:
        source_valid = torch.ones(
            (height, width), dtype=torch.bool, device=target_coordinates.device
        )
        target_valid = torch.ones(
            (target_coordinates.shape[0], height, width),
            dtype=torch.bool,
            device=target_coordinates.device,
        )
    else:
        source_hwc, source_valid = _unpad_normalized(
            source_coordinates.permute(1, 2, 0), paddings[0]
        )
        source_coordinates = source_hwc.permute(2, 0, 1)
        converted_targets = []
        converted_valid = []
        for target_index, padding in enumerate(paddings[1:]):
            target_hwc, valid = _unpad_normalized(
                target_coordinates[target_index].permute(1, 2, 0), padding
            )
            converted_targets.append(target_hwc.permute(2, 0, 1))
            converted_valid.append(valid)
        target_coordinates = torch.stack(converted_targets)
        target_valid = torch.stack(converted_valid)

    def sampling_geometry(index: int) -> ImageSamplingGeometry:
        shape = image_shapes[index]
        return ImageSamplingGeometry(
            original_height=shape.original_height,
            original_width=shape.original_width,
            storage_height=height,
            storage_width=width,
            square_size=shape.square_size if paddings is not None else None,
            padding_left=shape.left if paddings is not None else 0,
            padding_top=shape.top if paddings is not None else 0,
        )

    return DenseMultiViewCorrespondence(
        source_coordinates=source_coordinates,
        target_coordinates=target_coordinates,
        certainty_logits=certainty_logits,
        source_valid=source_valid,
        target_valid=target_valid,
        source_geometry=sampling_geometry(0),
        target_geometries=tuple(
            sampling_geometry(index) for index in range(1, len(paths))
        ),
    )


class MVRoMaRuntime:
    """Own the UFM → release → MV-RoMa lifecycle in one Python process."""

    def __init__(
        self,
        settings: MVRoMaSettings,
        device: torch.device | str,
        *,
        prematcher_factory: PrematcherFactory | None = None,
        track_provider: TrackProvider | None = None,
        model_factory: ModelFactory | None = None,
        dense_runner: DenseRunner | None = None,
    ) -> None:
        self.settings = settings
        self.device = torch.device(device)
        if prematcher_factory is None:
            self._prematcher_factory = partial(
                _default_prematcher_factory, settings=self.settings
            )
        else:
            self._prematcher_factory = prematcher_factory
        self._track_provider = track_provider or _default_track_provider
        self._model_factory = model_factory or _default_model_factory
        self._dense_runner = dense_runner or _default_dense_runner
        self._prematcher: Any = None
        self._model: torch.nn.Module | None = None
        self._bootstrapped = False

    def __enter__(self) -> "MVRoMaRuntime":
        self._bootstrap()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _bootstrap(self) -> None:
        if self._bootstrapped:
            return
        _validate_sources(self.settings)
        bootstrap_mvroma(self.settings.root)
        self._bootstrapped = True

    def _release_cuda(self) -> None:
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_tracks(
        self,
        groups: Sequence[InferenceGroup],
    ) -> list[PreparedInferenceGroup]:
        """Run UFM for every group, retaining only small CPU track-token inputs."""

        self._bootstrap()
        if self._model is not None:
            raise RuntimeError(
                "prepare_tracks must run before the MV-RoMa model is loaded"
            )
        prematcher = self._ensure_prematcher()
        prepared = []
        try:
            for group in tqdm(groups, desc="MV-RoMa UFM track preparation"):
                tracks = self._track_provider(
                    prematcher, group, self.settings, self.device
                )
                if not torch.is_tensor(tracks):
                    raise TypeError("track provider must return a torch.Tensor")
                if (
                    tracks.ndim != 3
                    or tracks.shape[0] != len(group.target_paths)
                    or tracks.shape[2] != 4
                ):
                    raise ValueError(
                        "track provider must return [target_count,track_count,4]"
                    )
                if tracks.shape[1] == 0:
                    print(
                        "Warning: UFM found no usable tracks for source "
                        f"{group.source_path.name}; skipping this camera group."
                    )
                prepared.append(
                    PreparedInferenceGroup(
                        group=group,
                        paired_tracks=tracks.detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                    )
                )
        finally:
            self.release_prematcher()
        return prepared

    def _ensure_prematcher(self) -> Any:
        self._bootstrap()
        if self._model is not None:
            raise RuntimeError("the UFM prematcher cannot be loaded after MV-RoMa")
        if self._prematcher is None:
            self._prematcher = self._prematcher_factory(
                self.settings.prematcher, self.device
            )
        return self._prematcher

    def estimate_visibility_overlap(
        self,
        image_paths: Sequence[Path],
        *,
        confidence_threshold: float,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Compute paper Eq.14 overlap with the already configured UFM model.

        This quality-oriented provider is optional because it is quadratic in
        the image count.  The prematcher stays alive and is reused immediately
        by ``prepare_tracks`` before being released.
        """

        if len(image_paths) < 2:
            raise ValueError("visibility overlap requires at least two images")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("overlap confidence threshold must be in [0,1]")
        if batch_size <= 0:
            raise ValueError("overlap batch_size must be positive")
        prematcher = self._ensure_prematcher()
        matcher_module = importlib.import_module("src.matchers.run_matcher_path")
        count = len(image_paths)
        overlap = np.zeros((count, count), dtype=np.float64)
        match_height, match_width = self.settings.prematch_resolution
        for source_index, source_path in enumerate(image_paths):
            targets = [index for index in range(count) if index != source_index]
            for start in range(0, len(targets), batch_size):
                target_indices = targets[start : start + batch_size]
                image_dict = {
                    "query_img_path": str(source_path),
                    "ref_img_paths": [str(image_paths[index]) for index in target_indices],
                }
                query, references, covisibility = matcher_module.run_match_multi_path(
                    image_dict,
                    match_W=match_width,
                    match_H=match_height,
                    matcher_model=prematcher,
                    matcher_name=self.settings.prematcher,
                    device=str(self.device),
                    batched=self.settings.prematch_batched,
                )
                ratios = (covisibility > confidence_threshold).float().mean(
                    dim=(-2, -1)
                )
                overlap[source_index, target_indices] = ratios.detach().cpu().numpy()
                del query, references, covisibility, ratios
        return overlap

    def release_prematcher(self) -> None:
        self._prematcher = None
        self._release_cuda()

    def _ensure_model(self) -> torch.nn.Module:
        self._bootstrap()
        if self._prematcher is not None:
            raise RuntimeError("release the UFM prematcher before loading MV-RoMa")
        if self._model is None:
            self._model = self._model_factory(self.settings, self.device)
        return self._model

    def predict_dense(
        self,
        prepared: PreparedInferenceGroup,
    ) -> DenseMultiViewCorrespondence:
        """Return the direct dense MV-RoMa output for one prepared group."""

        if prepared.paired_tracks.shape[1] == 0:
            raise ValueError("cannot run MV-RoMa with an empty track-token input")
        return self._dense_runner(
            self._ensure_model(), prepared, self.settings, self.device
        )

    def close(self) -> None:
        self._prematcher = None
        self._model = None
        self._release_cuda()
