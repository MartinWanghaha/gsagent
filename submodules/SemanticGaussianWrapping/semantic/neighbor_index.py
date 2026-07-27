"""Shared, refreshable nearest-neighbor selection for Gaussian geometry."""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class _SupportBucket:
    tree: Any
    indices: Any
    maximum_scale: float
    maximum_log_opacity: float


@dataclass(frozen=True)
class GaussianSupportAttributes:
    """Activated attributes for a sorted subset of live Gaussians.

    ``indices`` stay in the model's global index domain.  The remaining
    tensors are compact and live on the query device.  Keeping this view
    explicit prevents support reranking and surface evaluation from
    accidentally materializing activated attributes for the complete model.
    """

    indices: Tensor
    xyz: Tensor
    scaling: Tensor
    rotation: Tensor
    rotation_matrix: Tensor
    opacity: Tensor

    def local_indices(self, global_indices: Tensor) -> Tensor:
        """Map global Gaussian indices into this compact sorted view."""

        if global_indices.dtype != torch.long:
            global_indices = global_indices.long()
        compact_indices = self.indices.to(global_indices.device)
        # Callers construct this view from the exact union of their candidate
        # rows.  Avoid an equality check here: it would synchronize the GPU in
        # every support-ranking window, defeating the shared context.
        return torch.searchsorted(compact_indices, global_indices)


class GaussianNeighborIndex:
    """A detached spatial-index view of one live Gaussian model.

    The preferred backend is SciPy's CPU ``cKDTree``.  If SciPy is not
    installed, an exact torch scan is used with two-dimensional chunking so
    the temporary distance matrix remains bounded.  Only the integer neighbor
    selection is detached; consumers gather the live Gaussian tensors and
    therefore retain gradients for all continuous field computations.

    Registry topology edits replace the xyz parameter, so pointer/shape
    changes invalidate the index automatically.  In-place xyz optimization
    keeps the same pointer and intentionally requires an explicit ``refresh``.
    """

    _VALID_BACKENDS = {"auto", "scipy", "exact"}

    def __init__(
        self,
        gaussians: Any,
        *,
        backend: str = "auto",
        gaussian_chunk_size: int = 8192,
        query_chunk_size: int = 2048,
        max_distance_bytes: int = 64 * 1024 * 1024,
        support_candidate_budget: int = 2_048,
        support_routing_query_chunk: int = 8_192,
        scipy_workers: int = 4,
    ) -> None:
        if backend not in self._VALID_BACKENDS:
            raise ValueError("backend must be 'auto', 'scipy', or 'exact'")
        if (
            gaussian_chunk_size < 1
            or query_chunk_size < 1
            or max_distance_bytes < 1
            or support_candidate_budget < 1
            or support_routing_query_chunk < 1
            or scipy_workers == 0
            or scipy_workers < -1
        ):
            raise ValueError("neighbor-index chunk and memory limits must be positive")
        self.backend = backend
        self.gaussian_chunk_size = int(gaussian_chunk_size)
        self.query_chunk_size = int(query_chunk_size)
        self.max_distance_bytes = int(max_distance_bytes)
        self.support_candidate_budget = int(support_candidate_budget)
        # CPU cKDTree routing and GPU Mahalanobis reranking have different
        # workspace constraints.  Routing a larger row block amortizes one
        # tree query over many GPU-sized rerank windows without weakening the
        # latter's hard ``max_distance_bytes`` bound.
        self.support_routing_query_chunk = int(support_routing_query_chunk)
        self.scipy_workers = int(scipy_workers)
        self._lock = threading.Lock()
        self._tree = None
        self._support_buckets: tuple[_SupportBucket, ...] = ()
        self._signature = None
        self._active_backend: str | None = None
        self._support_context: ContextVar[GaussianSupportAttributes | None] = ContextVar(
            f"gaussian_support_context_{id(self)}",
            default=None,
        )
        self.set_gaussians(gaussians)

    def set_gaussians(self, gaussians: Any) -> None:
        """Retarget the index and invalidate every cached spatial structure."""

        object.__setattr__(self, "_gaussians_ref", weakref.ref(gaussians))
        self.invalidate()

    @property
    def gaussians(self) -> Any:
        value = self._gaussians_ref()
        if value is None:
            raise RuntimeError("the Gaussian model backing this neighbor index was released")
        return value

    @property
    def signature(self) -> tuple[int, tuple[int, ...], str, int | None] | None:
        return self._signature

    @property
    def active_backend(self) -> str | None:
        return self._active_backend

    @property
    def tree(self):
        """Expose the cached tree for diagnostics without making it mutable."""

        return self._tree

    def _current_signature(self) -> tuple[int, tuple[int, ...], str, int | None]:
        xyz = self.gaussians.get_xyz
        return int(xyz.data_ptr()), tuple(xyz.shape), xyz.device.type, xyz.device.index

    def _gather_attribute(
        self,
        indices: Tensor,
        *,
        raw_name: str,
        getter_name: str,
        reference: Tensor,
        activation: Any | None = None,
        detach: bool,
    ) -> Tensor:
        """Gather first, then activate registry-backed Gaussian attributes.

        The project model stores scale, rotation and opacity in their raw
        parameterizations.  Activating after ``index_select`` is exactly the
        same pointwise computation as activating the full tensor first, while
        its cost depends only on the candidate count.  Foreign/legacy models
        without the registry contract fall back to their already-activated
        getters once per query context.
        """

        gaussians = self.gaussians
        registry = getattr(gaussians, "registry", None)
        uses_raw = registry is not None and raw_name in registry
        if uses_raw:
            source = registry[raw_name]
        else:
            try:
                source = getattr(gaussians, getter_name)
            except AttributeError as exc:
                raise AttributeError(
                    f"support queries require Gaussian attribute {getter_name!r}"
                ) from exc
        model_indices = indices.to(device=source.device, dtype=torch.long)
        selected = source.index_select(0, model_indices)
        if detach:
            selected = selected.detach()
        if uses_raw and activation is not None:
            selected = activation(selected)
        return selected.to(device=reference.device, dtype=reference.dtype)

    def gather_support_attributes(
        self,
        indices: Tensor,
        reference: Tensor,
        *,
        detach: bool = False,
    ) -> GaussianSupportAttributes:
        """Gather and activate one sorted unique Gaussian candidate set.

        With ``detach=False`` the compact values retain gradients to the live
        registry parameters.  Neighbor selection uses ``detach=True`` because
        top-k routing is intentionally discrete.
        """

        if not isinstance(indices, Tensor) or indices.ndim != 1:
            raise ValueError("support indices must be a one-dimensional tensor")
        if not isinstance(reference, Tensor) or not reference.is_floating_point():
            raise TypeError("support attribute reference must be a floating-point tensor")
        xyz_source = self.gaussians.get_xyz
        normalized = indices.to(device=xyz_source.device, dtype=torch.long)
        if normalized.numel():
            if bool((normalized < 0).any()) or bool((normalized >= len(self.gaussians)).any()):
                raise IndexError("support Gaussian index is out of range")
            if normalized.numel() > 1 and bool((normalized[1:] <= normalized[:-1]).any()):
                raise ValueError("support indices must be sorted and unique")
        indices_on_query = normalized.to(reference.device)
        xyz = self._gather_attribute(
            normalized,
            raw_name="xyz",
            getter_name="get_xyz",
            reference=reference,
            detach=detach,
        )
        scaling = self._gather_attribute(
            normalized,
            raw_name="scaling",
            getter_name="get_scaling",
            reference=reference,
            activation=torch.exp,
            detach=detach,
        )
        rotation = self._gather_attribute(
            normalized,
            raw_name="rotation",
            getter_name="get_rotation",
            reference=reference,
            activation=lambda value: F.normalize(value, dim=-1, eps=1e-8),
            detach=detach,
        )
        opacity = self._gather_attribute(
            normalized,
            raw_name="opacity",
            getter_name="get_opacity",
            reference=reference,
            activation=torch.sigmoid,
            detach=detach,
        ).reshape(-1)
        return GaussianSupportAttributes(
            indices=indices_on_query,
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            rotation_matrix=self._rotation_matrix(rotation),
            opacity=opacity,
        )

    @contextmanager
    def _use_support_attributes(
        self,
        attributes: GaussianSupportAttributes,
    ) -> Iterator[None]:
        token = self._support_context.set(attributes)
        try:
            yield
        finally:
            self._support_context.reset(token)

    def invalidate(self) -> None:
        """Drop the cached index; the next query rebuilds it lazily."""

        with self._lock:
            self._tree = None
            self._support_buckets = ()
            self._signature = None
            self._active_backend = None

    @torch.no_grad()
    def refresh(self, force: bool = True) -> str:
        """Refresh positions and return the selected backend name."""

        signature = self._current_signature()
        if not force and self._active_backend is not None and signature == self._signature:
            return self._active_backend
        if self.backend == "exact":
            with self._lock:
                self._tree = None
                self._support_buckets = ()
                self._signature = signature
                self._active_backend = "exact"
            return "exact"
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            if self.backend == "scipy":
                raise RuntimeError("backend='scipy' requires scipy")
            with self._lock:
                self._tree = None
                self._support_buckets = ()
                self._signature = signature
                self._active_backend = "exact"
            return "exact"

        xyz = self.gaussians.get_xyz.detach().float().cpu().numpy().copy()
        tree = cKDTree(xyz)
        buckets: list[_SupportBucket] = []
        if xyz.shape[0] > 0:
            model_indices = torch.arange(xyz.shape[0], device=self.gaussians.get_xyz.device)
            reference = self.gaussians.get_xyz.detach()
            try:
                scaling = self._gather_attribute(
                    model_indices,
                    raw_name="scaling",
                    getter_name="get_scaling",
                    reference=reference,
                    activation=torch.exp,
                    detach=True,
                )
                opacity = self._gather_attribute(
                    model_indices,
                    raw_name="opacity",
                    getter_name="get_opacity",
                    reference=reference,
                    activation=torch.sigmoid,
                    detach=True,
                )
            except AttributeError:
                # Point-only consumers do not need anisotropic support
                # buckets.  ``query_support`` still reports the missing
                # attributes when it is actually requested.
                pass
            else:
                maximum_scale = scaling.float().amax(dim=-1).clamp_min(1e-12).cpu()
                log_opacity = opacity.float().reshape(-1).clamp_min(1e-12).log().cpu()
                bucket_ids = torch.floor(torch.log2(maximum_scale)).to(torch.int64)
                for bucket_id in torch.unique(bucket_ids, sorted=True).tolist():
                    selected = (bucket_ids == bucket_id).nonzero(as_tuple=False).flatten()
                    selected_numpy = selected.numpy()
                    buckets.append(
                        _SupportBucket(
                            tree=cKDTree(xyz[selected_numpy]),
                            indices=selected_numpy,
                            maximum_scale=float(maximum_scale[selected].max()),
                            maximum_log_opacity=float(log_opacity[selected].max()),
                        )
                    )
        with self._lock:
            self._tree = tree
            self._support_buckets = tuple(buckets)
            self._signature = signature
            self._active_backend = "scipy"
        return "scipy"

    def _ensure_current(self) -> str:
        signature = self._current_signature()
        if self._active_backend is None or signature != self._signature:
            return self.refresh(force=True)
        return self._active_backend

    @staticmethod
    def _validate_points(points: Tensor) -> None:
        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query points must be a torch.Tensor with shape [Q,3]")
        if not points.is_floating_point():
            raise TypeError("query points must be floating point")

    @staticmethod
    def _validate_k(k: int) -> int:
        k = int(k)
        if k < 1:
            raise ValueError("k must be positive")
        return k

    def _scipy_query(
        self,
        points: Tensor,
        k: int,
        query_indices: Tensor | None,
    ) -> Tensor:
        requested = k + (1 if query_indices is not None else 0)
        points_numpy = points.detach().float().cpu().numpy()
        _, indices = self._query_tree(self._tree, points_numpy, k=requested)
        candidates = torch.as_tensor(indices, dtype=torch.long, device=points.device)
        candidates = candidates.reshape(points.shape[0], requested)
        if query_indices is None:
            return candidates

        source = query_indices.to(device=points.device, dtype=torch.long).reshape(-1, 1)
        keep = candidates != source
        rank = keep.cumsum(dim=1)
        output = torch.empty((points.shape[0], k), dtype=torch.long, device=points.device)
        # cKDTree returns k+1 candidates, hence at least k survive exclusion.
        row, column = torch.where(keep & (rank <= k))
        output[row, rank[row, column] - 1] = candidates[row, column]
        return output

    def _query_tree(
        self,
        tree: Any,
        points: Any,
        *,
        k: int,
        distance_upper_bound: float | None = None,
    ) -> tuple[Any, Any]:
        """Query one SciPy tree with a bounded, configurable worker count."""

        kwargs: dict[str, Any] = {"k": int(k), "workers": self.scipy_workers}
        if distance_upper_bound is not None:
            kwargs["distance_upper_bound"] = float(distance_upper_bound)
        try:
            return tree.query(points, **kwargs)
        except TypeError:  # SciPy < 1.6 has no workers argument.
            kwargs.pop("workers")
            return tree.query(points, **kwargs)

    def _exact_query(
        self,
        points: Tensor,
        k: int,
        query_indices: Tensor | None,
    ) -> Tensor:
        xyz = self.gaussians.get_xyz.detach().to(device=points.device, dtype=points.dtype)
        count = points.shape[0]
        candidate_chunk = min(self.gaussian_chunk_size, max(xyz.shape[0], 1))
        bytes_per_distance = max(points.element_size(), 4)
        memory_queries = max(
            1,
            self.max_distance_bytes // max(bytes_per_distance * candidate_chunk, 1),
        )
        query_chunk = min(self.query_chunk_size, memory_queries)
        outputs: list[Tensor] = []
        for query_start in range(0, count, query_chunk):
            current = points[query_start : query_start + query_chunk].detach()
            current_source = (
                None
                if query_indices is None
                else query_indices[query_start : query_start + query_chunk].to(points.device)
            )
            best_distance = points.new_full((current.shape[0], k), float("inf"))
            best_index = torch.zeros((current.shape[0], k), dtype=torch.long, device=points.device)
            for start in range(0, xyz.shape[0], candidate_chunk):
                block = xyz[start : start + candidate_chunk]
                distance = torch.cdist(current, block).square()
                candidate_indices = torch.arange(start, start + block.shape[0], device=points.device)
                if current_source is not None:
                    distance.masked_fill_(
                        current_source[:, None] == candidate_indices[None, :],
                        float("inf"),
                    )
                expanded_indices = candidate_indices[None].expand(current.shape[0], -1)
                all_distance = torch.cat((best_distance, distance), dim=1)
                all_indices = torch.cat((best_index, expanded_indices), dim=1)
                best_distance, order = torch.topk(
                    all_distance,
                    k,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                best_index = all_indices.gather(1, order)
            outputs.append(best_index)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, k), dtype=torch.long, device=points.device)

    @staticmethod
    def _rotation_matrix(quaternion: Tensor) -> Tensor:
        quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=1e-8)
        w, x, y, z = quaternion.unbind(-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(quaternion.shape[:-1] + (3, 3))

    def _attributes_for_candidates(
        self,
        points: Tensor,
        candidates: Tensor,
    ) -> GaussianSupportAttributes:
        attributes = self._support_context.get()
        if attributes is not None:
            return attributes
        unique = torch.unique(candidates.reshape(-1), sorted=True)
        return self.gather_support_attributes(unique, points, detach=True)

    def _support_score_shared(
        self,
        points: Tensor,
        candidates: Tensor,
        density_scale: float,
    ) -> Tensor:
        attributes = self._attributes_for_candidates(points, candidates)
        local_indices = attributes.local_indices(candidates)
        centers = attributes.xyz.index_select(0, local_indices)
        scales = attributes.scaling.index_select(0, local_indices).clamp_min(1e-12)
        rotation = attributes.rotation_matrix.index_select(0, local_indices)
        alpha = attributes.opacity.index_select(0, local_indices).clamp_min(1e-12)
        delta = points[:, None, :] - centers[None, :, :]
        local = torch.einsum("cji,qcj->qci", rotation, delta)
        mahalanobis = (local / scales).square().sum(-1)
        return math.log(density_scale) + alpha.log()[None, :] - 0.5 * mahalanobis

    def _support_score_ragged(
        self,
        points: Tensor,
        candidates: Tensor,
        density_scale: float,
    ) -> Tensor:
        attributes = self._attributes_for_candidates(points, candidates)
        rows, width = candidates.shape
        local_indices = attributes.local_indices(candidates).reshape(-1)
        centers = attributes.xyz.index_select(0, local_indices).reshape(rows, width, 3)
        scales = attributes.scaling.index_select(0, local_indices).reshape(rows, width, 3)
        scales = scales.clamp_min(1e-12)
        rotation = attributes.rotation_matrix.index_select(0, local_indices).reshape(rows, width, 3, 3)
        alpha = attributes.opacity.index_select(0, local_indices).reshape(rows, width)
        delta = points[:, None, :] - centers
        local = torch.einsum("qlji,qlj->qli", rotation, delta)
        mahalanobis = (local / scales).square().sum(-1)
        return math.log(density_scale) + alpha.clamp_min(1e-12).log() - 0.5 * mahalanobis

    @staticmethod
    def _length_bucket(length: int) -> int:
        return 1 << max(length - 1, 0).bit_length()

    def _batch_ragged_support(
        self,
        points: Tensor,
        candidate_rows: list[list[int]],
        k: int,
        density_scale: float,
    ) -> Tensor:
        """Batch ragged candidate rows without per-query tensor kernels."""

        groups: dict[int, list[int]] = {}
        for row_index, candidates in enumerate(candidate_rows):
            groups.setdefault(self._length_bucket(len(candidates)), []).append(row_index)

        output = torch.empty((points.shape[0], k), dtype=torch.long, device=points.device)
        element_size = max(points.element_size(), 4)
        maximum_width = max(1, self.max_distance_bytes // max(32 * element_size, 1))
        for bucket_width, row_indices in groups.items():
            window_width = min(bucket_width, maximum_width)
            rows_per_batch = max(
                1,
                self.max_distance_bytes // max(32 * element_size * window_width, 1),
            )
            rows_per_batch = min(rows_per_batch, self.query_chunk_size)
            for row_start in range(0, len(row_indices), rows_per_batch):
                current_rows = row_indices[row_start : row_start + rows_per_batch]
                row_tensor = torch.as_tensor(current_rows, dtype=torch.long, device=points.device)
                current_points = points.index_select(0, row_tensor)
                best_score = points.new_full((len(current_rows), k), -float("inf"))
                best_index = torch.zeros((len(current_rows), k), dtype=torch.long, device=points.device)
                longest = max(len(candidate_rows[row]) for row in current_rows)
                for candidate_start in range(0, longest, window_width):
                    slices = [
                        candidate_rows[row][candidate_start : candidate_start + window_width]
                        for row in current_rows
                    ]
                    lengths = [len(values) for values in slices]
                    padded = [values + [0] * (window_width - len(values)) for values in slices]
                    candidates = torch.as_tensor(padded, dtype=torch.long, device=points.device)
                    valid_length = torch.as_tensor(lengths, dtype=torch.long, device=points.device)
                    valid = torch.arange(window_width, device=points.device)[None] < valid_length[:, None]
                    score = self._support_score_ragged(current_points, candidates, density_scale)
                    score.masked_fill_(~valid, -float("inf"))
                    all_score = torch.cat((best_score, score), dim=1)
                    all_index = torch.cat((best_index, candidates), dim=1)
                    best_score, order = torch.topk(all_score, k, dim=1, largest=True, sorted=True)
                    best_index = all_index.gather(1, order)
                output.index_copy_(0, row_tensor, best_index)
        return output

    @staticmethod
    def _allocate_bucket_quotas(
        capacities: list[int],
        budget: int,
    ) -> list[int]:
        """Distribute a fixed shortlist budget across scale buckets."""

        quotas = [0] * len(capacities)
        remaining = max(int(budget), 0)
        active = [index for index, capacity in enumerate(capacities) if capacity > 0]
        while remaining > 0 and active:
            share = max(remaining // len(active), 1)
            consumed = 0
            next_active = []
            for index in active:
                available = capacities[index] - quotas[index]
                addition = min(share, available, remaining - consumed)
                quotas[index] += addition
                consumed += addition
                if quotas[index] < capacities[index]:
                    next_active.append(index)
                if consumed == remaining:
                    break
            if consumed == 0:
                break
            remaining -= consumed
            active = next_active
        return quotas

    def _batch_dense_support(
        self,
        points: Tensor,
        candidates: Tensor,
        valid: Tensor,
        k: int,
        density_scale: float,
    ) -> Tensor:
        """Exact support reranking over a fixed-width candidate shortlist."""

        width = candidates.shape[1]
        rows_per_batch = self._support_workspace_rows(points, width)
        output = torch.empty((points.shape[0], k), dtype=torch.long, device=points.device)
        for start in range(0, points.shape[0], rows_per_batch):
            stop = min(start + rows_per_batch, points.shape[0])
            current_candidates = candidates[start:stop]
            current_valid = valid[start:stop]
            # Sorting makes duplicate removal deterministic. Invalid padding
            # sorts after every real model index and is never gathered.
            sort_key = torch.where(
                current_valid,
                current_candidates,
                torch.full_like(current_candidates, len(self.gaussians)),
            )
            sort_key, order = sort_key.sort(dim=1)
            current_valid = current_valid.gather(1, order)
            safe_candidates = torch.where(
                current_valid,
                sort_key,
                sort_key[:, :1].expand_as(sort_key),
            )
            duplicate = torch.zeros_like(current_valid)
            duplicate[:, 1:] = (
                (sort_key[:, 1:] == sort_key[:, :-1])
                & current_valid[:, 1:]
                & current_valid[:, :-1]
            )
            current_valid &= ~duplicate
            score = self._support_score_ragged(
                points[start:stop],
                safe_candidates,
                density_scale,
            )
            score.masked_fill_(~current_valid, -float("inf"))
            _, selected = torch.topk(score, k, dim=1, largest=True, sorted=True)
            output[start:stop] = safe_candidates.gather(1, selected)
        return output

    def _support_workspace_rows(self, points: Tensor, width: int) -> int:
        """Return a hard query-row bound for shortlist construction/reranking.

        Candidate construction temporarily coexists with validity masks and
        the integer sort/deduplication workspace.  The same deliberately
        conservative 32-value estimate is used by the exact support path, so
        ``max_distance_bytes`` bounds the whole routing block rather than only
        the final Mahalanobis score tensor.
        """

        element_size = max(points.element_size(), 4)
        bytes_per_row = 32 * element_size * max(int(width), 1)
        if bytes_per_row > self.max_distance_bytes:
            raise ValueError(
                "max_distance_bytes is too small for one support shortlist row: "
                f"need at least {bytes_per_row} bytes for width {width}"
            )
        return min(
            self.query_chunk_size,
            max(self.max_distance_bytes // bytes_per_row, 1),
        )

    def _scipy_support_candidates_cpu(
        self,
        points_numpy: np.ndarray,
        k: int,
        active: list[tuple[_SupportBucket, float]],
        quotas: list[int],
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build one fixed-width candidate block entirely in CPU memory.

        Each scale tree sees the complete routing block once.  Candidate and
        validity arrays cross to the query device only in GPU-workspace-sized
        slices during exact reranking, avoiding one H2D transfer per bucket.
        """

        _, center = self._query_tree(self._tree, points_numpy, k=k)
        rows = int(points_numpy.shape[0])
        center = np.asarray(center, dtype=np.int64).reshape(rows, k)
        candidates = np.zeros((rows, width), dtype=np.int64)
        valid = np.zeros((rows, width), dtype=np.bool_)
        candidates[:, :k] = center
        valid[:, :k] = True
        offset = k
        for (bucket, radius), quota in zip(active, quotas):
            if quota == 0:
                continue
            _, local = self._query_tree(
                bucket.tree,
                points_numpy,
                k=quota,
                distance_upper_bound=radius,
            )
            local = np.asarray(local, dtype=np.int64).reshape(rows, quota)
            valid_numpy = local < len(bucket.indices)
            safe_local = local.copy()
            safe_local[~valid_numpy] = 0
            global_indices = bucket.indices[safe_local]
            stop = offset + quota
            candidates[:, offset:stop] = global_indices
            valid[:, offset:stop] = valid_numpy
            offset = stop
        return candidates, valid

    def _scipy_support_query(
        self,
        points: Tensor,
        k: int,
        density_scale: float,
        minimum_log_support: float,
    ) -> Tensor:
        # Center neighbors guarantee k valid outputs. Scale buckets contribute
        # a fixed-budget, multi-scale Euclidean shortlist inside the same
        # conservative support radii. Exact anisotropic Mahalanobis ranking is
        # then applied to that shortlist. Unlike query_ball_point, work and
        # memory cannot explode because of one oversized Gaussian or a dense
        # scale bucket.
        active: list[tuple[_SupportBucket, float]] = []
        for bucket in self._support_buckets:
            log_peak = math.log(density_scale) + bucket.maximum_log_opacity
            margin = 2.0 * (log_peak - minimum_log_support)
            if margin < 0.0:
                continue
            radius = bucket.maximum_scale * math.sqrt(margin)
            active.append((bucket, radius))

        width = max(k, self.support_candidate_budget)
        quotas = self._allocate_bucket_quotas(
            [len(bucket.indices) for bucket, _ in active],
            width - k,
        )
        rerank_rows = self._support_workspace_rows(points, width)
        routing_rows = self.support_routing_query_chunk
        output = torch.empty(
            (points.shape[0], k),
            dtype=torch.long,
            device=points.device,
        )
        for route_start in range(0, points.shape[0], routing_rows):
            route_stop = min(route_start + routing_rows, points.shape[0])
            routed_points = points[route_start:route_stop]
            # This is the only query-device to CPU copy for the entire routing
            # block. The old path repeated it for the center tree and again for
            # every small GPU rerank window.
            points_numpy = routed_points.detach().float().cpu().numpy()
            candidate_numpy, valid_numpy = self._scipy_support_candidates_cpu(
                points_numpy,
                k,
                active,
                quotas,
                width,
            )
            for local_start in range(0, routed_points.shape[0], rerank_rows):
                local_stop = min(local_start + rerank_rows, routed_points.shape[0])
                current = routed_points[local_start:local_stop]
                # Transfer exactly one candidates slice and one validity slice
                # per rerank window. The full CPU routing block never occupies
                # query-device memory, preserving the hard workspace bound.
                candidates = torch.as_tensor(
                    np.ascontiguousarray(candidate_numpy[local_start:local_stop]),
                    dtype=torch.long,
                    device=points.device,
                )
                valid = torch.as_tensor(
                    np.ascontiguousarray(valid_numpy[local_start:local_stop]),
                    dtype=torch.bool,
                    device=points.device,
                )
                unique_candidates = torch.unique(candidates[valid], sorted=True)
                attributes = self.gather_support_attributes(
                    unique_candidates,
                    current,
                    detach=True,
                )
                with self._use_support_attributes(attributes):
                    output_start = route_start + local_start
                    output_stop = route_start + local_stop
                    output[output_start:output_stop] = self._batch_dense_support(
                        current,
                        candidates,
                        valid,
                        k,
                        density_scale,
                    )
        return output

    def _exact_support_query(self, points: Tensor, k: int, density_scale: float) -> Tensor:
        gaussian_count = len(self.gaussians)
        bytes_per_value = max(points.element_size(), 4)
        maximum_candidates = max(
            1,
            self.max_distance_bytes // max(32 * bytes_per_value, 1),
        )
        candidate_chunk = min(self.gaussian_chunk_size, gaussian_count, maximum_candidates)
        # Batched delta/local/score tensors dominate pairwise working memory.
        memory_queries = max(
            1,
            self.max_distance_bytes // max(32 * bytes_per_value * candidate_chunk, 1),
        )
        query_chunk = min(self.query_chunk_size, memory_queries)
        best_score = points.new_full((points.shape[0], k), -float("inf"))
        best_index = torch.zeros(
            (points.shape[0], k),
            dtype=torch.long,
            device=points.device,
        )
        registry = getattr(self.gaussians, "registry", None)
        can_gather_raw_blocks = registry is not None and all(
            name in registry for name in ("xyz", "scaling", "rotation", "opacity")
        )
        shared_attributes = None
        if not can_gather_raw_blocks:
            # A foreign model exposes activated full-tensor getters rather than
            # the registry's raw per-candidate storage. Evaluate those getters
            # once to preserve its API contract; project models take the
            # bounded gather-first branch below.
            shared_attributes = self.gather_support_attributes(
                torch.arange(
                    gaussian_count,
                    dtype=torch.long,
                    device=points.device,
                ),
                points,
                detach=True,
            )
        for candidate_start in range(0, gaussian_count, candidate_chunk):
            candidates = torch.arange(
                candidate_start,
                min(candidate_start + candidate_chunk, gaussian_count),
                device=points.device,
            )
            # Exact fallback still gathers before activating, but does so one
            # bounded candidate block at a time and shares that compact block
            # across all query chunks. Materializing a full second activated
            # copy of a multi-million Gaussian model can otherwise OOM.
            attributes = shared_attributes or self.gather_support_attributes(
                candidates,
                points,
                detach=True,
            )
            with self._use_support_attributes(attributes):
                for query_start in range(0, points.shape[0], query_chunk):
                    stop = min(query_start + query_chunk, points.shape[0])
                    current = points[query_start:stop]
                    score = self._support_score_shared(
                        current,
                        candidates,
                        density_scale,
                    )
                    current_best_score = best_score[query_start:stop]
                    current_best_index = best_index[query_start:stop]
                    expanded = candidates[None].expand(current.shape[0], -1)
                    all_score = torch.cat((current_best_score, score), dim=1)
                    all_index = torch.cat((current_best_index, expanded), dim=1)
                    selected_score, order = torch.topk(
                        all_score,
                        k,
                        dim=1,
                        largest=True,
                        sorted=True,
                    )
                    best_score[query_start:stop] = selected_score
                    best_index[query_start:stop] = all_index.gather(1, order)
        return best_index

    @torch.no_grad()
    def query_points(self, points: Tensor, k: int) -> Tensor:
        """Return nearest Gaussian indices for arbitrary query points."""

        self._validate_points(points)
        k = min(self._validate_k(k), len(self.gaussians))
        if points.shape[0] == 0 or k == 0:
            return torch.empty((points.shape[0], 0), dtype=torch.long, device=points.device)
        backend = self._ensure_current()
        if backend == "scipy":
            return self._scipy_query(points, k, None)
        return self._exact_query(points, k, None)

    @torch.no_grad()
    def query_support(
        self,
        points: Tensor,
        k: int,
        *,
        density_scale: float = 8.0,
        minimum_log_support: float = -12.0,
    ) -> Tensor:
        """Select top anisotropic Gaussian support, not nearest centers.

        The cKDTree path uses log2 maximum-scale buckets to produce a
        conservative candidate superset. Candidates are then ranked with the
        live rotation, anisotropic scale, opacity, and Mahalanobis distance.
        The exact fallback performs the same ranking in bounded chunks.
        """

        self._validate_points(points)
        if density_scale <= 0.0 or not math.isfinite(density_scale):
            raise ValueError("density_scale must be finite and positive")
        if not math.isfinite(minimum_log_support):
            raise ValueError("minimum_log_support must be finite")
        k = min(self._validate_k(k), len(self.gaussians))
        if points.shape[0] == 0 or k == 0:
            return torch.empty((points.shape[0], 0), dtype=torch.long, device=points.device)
        backend = self._ensure_current()
        if backend == "scipy":
            return self._scipy_support_query(points, k, density_scale, minimum_log_support)
        return self._exact_support_query(points, k, density_scale)

    @torch.no_grad()
    def query_indices(
        self,
        indices: Tensor,
        k: int,
        *,
        exclude_self: bool = True,
    ) -> Tensor:
        """Query Gaussian centers, optionally excluding each source index."""

        xyz = self.gaussians.get_xyz
        indices = torch.as_tensor(indices, device=xyz.device)
        if indices.dtype == torch.bool:
            if indices.shape != (len(self.gaussians),):
                raise ValueError(f"boolean indices must have shape [{len(self.gaussians)}]")
            indices = indices.nonzero(as_tuple=False).flatten()
        else:
            indices = indices.long().flatten()
        if indices.numel() and (bool((indices < 0).any()) or bool((indices >= len(self.gaussians)).any())):
            raise IndexError("Gaussian query index is out of range")
        available = len(self.gaussians) - (1 if exclude_self else 0)
        k = min(self._validate_k(k), max(available, 0))
        if indices.numel() == 0 or k == 0:
            return torch.empty((indices.numel(), 0), dtype=torch.long, device=indices.device)
        points = xyz.detach().index_select(0, indices)
        backend = self._ensure_current()
        source = indices if exclude_self else None
        if backend == "scipy":
            return self._scipy_query(points, k, source)
        return self._exact_query(points, k, source)


__all__ = ["GaussianNeighborIndex", "GaussianSupportAttributes"]
