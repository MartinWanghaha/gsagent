"""Strict adapter for the unified :class:`SemanticSurfaceField` contract."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Callable, Optional

import numpy as np

from .types import RegionOwnershipSamples, SurfaceSamples


FIELD_KEYS = (
    "occupancy",
    "sdf",
    "normal",
    "semantic",
    "geometry_posterior",
    "uncertainty",
)

class FieldContractError(RuntimeError):
    """Raised when a field does not implement the documented query contract."""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    values = {key: getattr(result, key) for key in FIELD_KEYS if hasattr(result, key)}
    if values:
        return values
    raise FieldContractError("SemanticSurfaceField.query must return a mapping")


def _strict_result_mapping(result: Any, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if isinstance(result, Mapping):
        mapping = dict(result)
    else:
        mapping = {key: getattr(result, key) for key in keys if hasattr(result, key)}
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise FieldContractError(f"{name} is missing: " + ", ".join(missing))
    return mapping


def _validate_region_ids(region_ids: Any) -> np.ndarray:
    raw = np.asarray(region_ids)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("region_ids must be a one-dimensional integer array")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if not len(result):
        raise ValueError("at least one foreground region ID is required")
    if np.any(result <= 0):
        raise ValueError("region_ids must contain foreground IDs greater than zero")
    if np.any(np.diff(result) <= 0):
        raise ValueError("region_ids must be sorted and unique")
    return result


class SurfaceFieldAdapter:
    """Chunked, no-grad NumPy access to a torch-backed semantic surface field."""

    def __init__(
        self,
        field: Any,
        *,
        device: Optional[str] = None,
        chunk_size: int = 262_144,
        input_type: str = "auto",
        decode_semantics: bool = True,
    ) -> None:
        if not hasattr(field, "query"):
            raise FieldContractError("field must define query(points)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if input_type not in {"auto", "torch", "numpy"}:
            raise ValueError("input_type must be auto, torch, or numpy")
        if not isinstance(decode_semantics, bool):
            raise TypeError("decode_semantics must be boolean")
        self.field = field
        self.chunk_size = int(chunk_size)
        self.input_type = input_type
        self.decode_semantics = decode_semantics
        self.device = device or self._infer_device()
        self.last_semantic_decoded = False

    def geometry_view(self) -> "SurfaceFieldAdapter":
        """Return the explicit geometry-only view used to build shared topology."""

        return SurfaceFieldAdapter(
            self.field,
            device=self.device,
            chunk_size=self.chunk_size,
            input_type=self.input_type,
            decode_semantics=False,
        )

    def _infer_device(self) -> str:
        explicit = getattr(self.field, "device", None)
        if explicit is not None:
            return str(explicit)
        parameters = getattr(self.field, "parameters", None)
        if callable(parameters):
            try:
                return str(next(parameters()).device)
            except (StopIteration, TypeError):
                pass
        buffers = getattr(self.field, "buffers", None)
        if callable(buffers):
            try:
                return str(next(buffers()).device)
            except (StopIteration, TypeError):
                pass
        return "cpu"

    def _uses_torch(self) -> bool:
        if self.input_type == "torch":
            return True
        if self.input_type == "numpy":
            return False
        try:
            import torch

            return isinstance(self.field, torch.nn.Module)
        except ImportError:
            return False

    def _query_once(self, points: np.ndarray) -> dict[str, np.ndarray]:
        use_torch = self._uses_torch()
        if use_torch:
            try:
                import torch
            except ImportError as error:
                raise FieldContractError(
                    "torch is required to query this SemanticSurfaceField"
                ) from error
            tensor = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            context = torch.no_grad()
            query_input: Any = tensor
        else:
            context = nullcontext()
            query_input = points

        with context:
            geometry_query = getattr(self.field, "query_geometry", None)
            query = (
                geometry_query
                if not self.decode_semantics and callable(geometry_query)
                else self.field.query
            )
            try:
                field_chunk = int(getattr(self.field, "query_chunk_size", len(points)))
                result = query(query_input, chunk_size=min(len(points), field_chunk))
            except TypeError as first_error:
                try:
                    result = query(query_input)
                except TypeError:
                    raise first_error

        mapping = _result_mapping(result)
        missing = [key for key in FIELD_KEYS if key not in mapping]
        if missing:
            raise FieldContractError(
                "SemanticSurfaceField.query is missing: " + ", ".join(missing)
            )
        return {key: _to_numpy(mapping[key]) for key in FIELD_KEYS}

    def query(
        self,
        points: np.ndarray,
        *,
        chunk_size: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> SurfaceSamples:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [P, 3]")
        if not len(points):
            raise ValueError("cannot query an empty point array")

        step = int(chunk_size or self.chunk_size)
        chunks: list[SurfaceSamples] = []
        for start in range(0, len(points), step):
            chunk_points = np.ascontiguousarray(points[start : start + step])
            values = self._query_once(chunk_points)
            try:
                samples = SurfaceSamples(points=chunk_points, **values)
            except ValueError as error:
                raise FieldContractError(str(error)) from error
            arrays = (
                samples.occupancy,
                samples.sdf,
                samples.normal,
                samples.semantic,
                samples.geometry_posterior,
                samples.uncertainty,
            )
            if any(not np.all(np.isfinite(array)) for array in arrays):
                raise FieldContractError("field query returned NaN or infinite values")
            normal_length = np.linalg.norm(samples.normal, axis=1, keepdims=True)
            samples.normal = samples.normal / np.maximum(normal_length, 1e-8)
            chunks.append(samples)
            if progress is not None:
                progress(min(start + len(chunk_points), len(points)), len(points))
        return chunks[0] if len(chunks) == 1 else SurfaceSamples.concatenate(chunks)

    def query_region_ownership(
        self,
        points: np.ndarray,
        *,
        region_ids: Any,
        chunk_size: Optional[int] = None,
    ) -> RegionOwnershipSamples:
        """Query all requested owners without constructing a dense region field."""

        ownership_query = getattr(self.field, "query_region_ownership", None)
        if not callable(ownership_query):
            raise FieldContractError(
                "field must define query_region_ownership(points, *, region_ids)"
            )
        if not self._uses_torch():
            raise FieldContractError(
                "query_region_ownership requires a torch-backed field"
            )
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [P, 3]")
        if not len(points):
            raise ValueError("cannot query an empty point array")
        requested_regions = _validate_region_ids(region_ids)
        step = self.chunk_size if chunk_size is None else int(chunk_size)
        if step < 1:
            raise ValueError("chunk_size must be positive")

        try:
            import torch
        except ImportError as error:
            raise FieldContractError(
                "torch is required for query_region_ownership"
            ) from error
        query_points = torch.as_tensor(
            points,
            dtype=torch.float32,
            device=self.device,
        )
        query_regions = torch.as_tensor(
            requested_regions,
            dtype=torch.long,
            device=self.device,
        )
        field_chunk = int(getattr(self.field, "query_chunk_size", step))
        if field_chunk < 1:
            raise FieldContractError("field query_chunk_size must be positive")
        with torch.no_grad():
            result = ownership_query(
                query_points,
                region_ids=query_regions,
                chunk_size=min(step, field_chunk),
            )
        mapping = _strict_result_mapping(
            result,
            ("requested_region_ids", "region_id", "confidence", "valid"),
            "region ownership",
        )
        returned_regions = _to_numpy(mapping["requested_region_ids"])
        if returned_regions.dtype.kind not in "iu" or not np.array_equal(
            returned_regions.astype(np.int64, copy=False),
            requested_regions,
        ):
            raise FieldContractError(
                "ownership requested_region_ids do not match the request"
            )
        valid = _to_numpy(mapping["valid"])
        if valid.dtype != np.bool_:
            raise FieldContractError("region ownership valid must use boolean dtype")
        try:
            return RegionOwnershipSamples(
                requested_region_ids=returned_regions,
                region_id=_to_numpy(mapping["region_id"]),
                confidence=_to_numpy(mapping["confidence"]),
                valid=valid,
            )
        except (TypeError, ValueError) as error:
            raise FieldContractError(str(error)) from error

    def semantic_ids(
        self,
        semantic: np.ndarray,
        decoder: Optional[Callable[[Any], Any]] = None,
    ) -> Optional[np.ndarray]:
        """Decode instance IDs, preferring a field-owned Gaga decoder when present."""
        semantic = np.asarray(semantic, dtype=np.float32)
        if not self.decode_semantics:
            self.last_semantic_decoded = False
            return None
        selected = decoder
        if selected is None:
            direct_decoder = getattr(self.field, "semantic_decoder", None)
            gaussian_decoder = getattr(
                getattr(self.field, "gaussians", None), "semantic_decoder", None
            )
            if callable(direct_decoder) or callable(gaussian_decoder):
                wrapper = getattr(self.field, "decode_semantic", None)
                selected = wrapper if callable(wrapper) else direct_decoder or gaussian_decoder
            elif not hasattr(self.field, "semantic_decoder"):
                candidate = getattr(self.field, "decode_semantic", None)
                if callable(candidate):
                    selected = candidate
        if selected is None:
            self.last_semantic_decoded = False
            raise FieldContractError(
                "discrete semantic mesh output requires a scene semantic decoder"
            )

        use_torch = self._uses_torch() or hasattr(selected, "parameters")
        if use_torch:
            try:
                import torch
            except ImportError as error:
                raise FieldContractError("semantic decoder requires torch") from error
            decoder_input: Any = torch.as_tensor(
                semantic, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                decoded = selected(decoder_input)
        else:
            decoded = selected(semantic)
        if decoded is None:
            self.last_semantic_decoded = False
            raise FieldContractError("semantic decoder returned no labels")
        decoded = _to_numpy(decoded)
        if decoded.ndim == 2:
            # This adapter populates the final TriangleMesh semantic_id output;
            # internal surface routing remains soft and never consumes it.
            decoded = np.argmax(decoded, axis=1)
        decoded = np.asarray(decoded).reshape(-1)
        if len(decoded) != len(semantic):
            raise FieldContractError("semantic decoder must return [P] IDs or [P,C] logits")
        self.last_semantic_decoded = True
        return decoded.astype(np.int32)


def as_field_adapter(field: Any, **kwargs: Any) -> SurfaceFieldAdapter:
    return field if isinstance(field, SurfaceFieldAdapter) else SurfaceFieldAdapter(field, **kwargs)
