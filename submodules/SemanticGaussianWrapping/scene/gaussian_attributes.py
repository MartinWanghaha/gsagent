"""Atomic lifecycle management for dynamically extensible Gaussian state."""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    trailing_shape: tuple[int, ...]
    trainable: bool = True
    optimizer_group: str | None = None
    role: str = "render"
    persistent: bool = True
    ply_prefix: str | None = None
    storage_order: str = "flat"


def _as_index(selection: Tensor | Sequence[int], size: int, device: torch.device) -> Tensor:
    result = torch.as_tensor(selection, device=device)
    if result.dtype == torch.bool:
        if result.ndim != 1 or result.numel() != size:
            raise ValueError(f"boolean selection must have shape [{size}]")
        result = result.nonzero(as_tuple=False).flatten()
    else:
        result = result.long().flatten()
    if result.numel() and (result.min() < 0 or result.max() >= size):
        raise IndexError("Gaussian index lies outside the registry")
    return result


class GaussianAttributeRegistry(nn.Module):
    """Own every tensor whose first dimension is the Gaussian topology.

    Replacing topology through :meth:`mutate` also replaces optimizer
    parameter references and transforms first-order/second-order state in the
    same transaction.  Newly cloned/split rows intentionally start with zero
    optimizer momentum.
    """

    FORMAT_VERSION = 2

    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.ParameterDict()
        self._specs: dict[str, AttributeSpec] = {}
        self._buffer_names: dict[str, str] = {}
        self.optimizer: torch.optim.Optimizer | None = None

    def __len__(self) -> int:
        if not self._specs:
            return 0
        return int(self[next(iter(self._specs))].shape[0])

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def specs(self) -> Mapping[str, AttributeSpec]:
        return dict(self._specs)

    @property
    def device(self) -> torch.device:
        return self[next(iter(self._specs))].device if self._specs else torch.device("cpu")

    def __getitem__(self, name: str) -> Tensor:
        spec = self._specs[name]
        if spec.trainable:
            return self.trainable[name]
        return getattr(self, self._buffer_names[name])

    def register(
        self,
        name: str,
        value: Tensor,
        *,
        trainable: bool = True,
        optimizer_group: str | None = None,
        role: str = "render",
        persistent: bool = True,
        ply_prefix: str | None = None,
        storage_order: str = "flat",
    ) -> Tensor:
        if not _VALID_NAME.match(name):
            raise ValueError(f"invalid Gaussian attribute name {name!r}")
        if name in self._specs:
            raise KeyError(f"Gaussian attribute {name!r} is already registered")
        value = torch.as_tensor(value)
        if value.ndim < 1:
            raise ValueError("Gaussian attributes require a leading topology dimension")
        if self._specs and value.shape[0] != len(self):
            raise ValueError(f"attribute {name!r} has {value.shape[0]} rows, expected {len(self)}")
        if self._specs and value.device != self.device:
            raise ValueError(f"attribute {name!r} is on {value.device}, registry is on {self.device}")
        if trainable and not (value.is_floating_point() or value.is_complex()):
            raise TypeError("trainable Gaussian attributes must be floating point")
        spec = AttributeSpec(
            name=name,
            trailing_shape=tuple(value.shape[1:]),
            trainable=trainable,
            optimizer_group=optimizer_group or (name if trainable else None),
            role=role,
            persistent=persistent,
            ply_prefix=ply_prefix,
            storage_order=storage_order,
        )
        self._specs[name] = spec
        if trainable:
            parameter = value if isinstance(value, nn.Parameter) else nn.Parameter(value)
            self.trainable[name] = parameter
            if self.optimizer is not None:
                matching = [
                    group
                    for group in self.optimizer.param_groups
                    if group.get("name") == spec.optimizer_group
                ]
                if matching:
                    matching[0]["params"].append(parameter)
                else:
                    self.optimizer.add_param_group(
                        {"params": [parameter], "name": spec.optimizer_group, "lr": 0.0}
                    )
        else:
            buffer_name = f"attribute_buffer_{name}"
            self._buffer_names[name] = buffer_name
            self.register_buffer(buffer_name, value, persistent=persistent)
        return self[name]

    def unregister(self, name: str) -> Tensor:
        if self.optimizer is not None:
            raise RuntimeError("unbind the optimizer before unregistering an attribute")
        value = self[name]
        spec = self._specs.pop(name)
        if spec.trainable:
            del self.trainable[name]
        else:
            delattr(self, self._buffer_names.pop(name))
        return value

    def bind_optimizer(self, optimizer: torch.optim.Optimizer | None) -> None:
        if optimizer is not None:
            registered = {id(self[name]) for name, spec in self._specs.items() if spec.trainable}
            optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            missing = registered - optimized
            if missing:
                names = [name for name, spec in self._specs.items() if spec.trainable and id(self[name]) in missing]
                raise ValueError(f"optimizer is missing registered parameters: {names}")
        self.optimizer = optimizer

    def named_attributes(self) -> Iterator[tuple[str, Tensor]]:
        for name in self._specs:
            yield name, self[name]

    def parameters_by_role(self, role: str) -> Iterator[nn.Parameter]:
        for name, spec in self._specs.items():
            if spec.trainable and spec.role == role:
                yield self.trainable[name]

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, spec in self._specs.items():
            if spec.trainable:
                groups.setdefault(spec.role, []).append(self.trainable[name])
        return groups

    def replace(
        self,
        name: str,
        value: Tensor,
        *,
        reset_optimizer_moments: bool = False,
    ) -> Tensor:
        """Replace one attribute without changing topology or sibling identities.

        Topology mutations necessarily recreate every topology-shaped parameter.
        Scalar maintenance operations such as opacity reset do not.  Keeping the
        two transactions separate prevents cached gradient-routing views from
        becoming stale and avoids copying all Gaussian attributes just to update
        one of them.
        """

        if name not in self._specs:
            raise KeyError(name)
        spec = self._specs[name]
        old = self[name]
        replacement = torch.as_tensor(value, device=old.device, dtype=old.dtype)
        if tuple(replacement.shape) != tuple(old.shape):
            raise ValueError(
                f"replacement {name!r} has shape {tuple(replacement.shape)}, "
                f"expected {tuple(old.shape)}"
            )
        replacement = replacement.contiguous()
        if spec.trainable and self.optimizer is not None:
            matches = [
                group
                for group in self.optimizer.param_groups
                if any(candidate is old for candidate in group["params"])
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"optimizer must contain attribute {name!r} exactly once; found {len(matches)}"
                )

        # Commit only after shape, device and optimizer ownership validation.
        # A same-shape maintenance update does not require a new Parameter:
        # preserving identity keeps trainer/Pareto views and optimizer groups
        # valid without any coordination outside the registry.
        with torch.no_grad():
            old.copy_(replacement)
            if spec.trainable and self.optimizer is not None and reset_optimizer_moments:
                for state_value in self.optimizer.state.get(old, {}).values():
                    if isinstance(state_value, Tensor) and state_value.ndim > 0:
                        state_value.zero_()
        return old

    def mutate(
        self,
        source_indices: Tensor | Sequence[int],
        *,
        overrides: Mapping[str, Tensor] | None = None,
        fresh_mask: Tensor | Sequence[bool] | None = None,
    ) -> dict[str, Tensor]:
        """Atomically replace topology from an old-to-new source index map.

        ``source_indices[j]`` identifies the old row used to initialize new row
        ``j``.  ``overrides`` may replace complete new attributes.  Rows marked
        by ``fresh_mask`` receive zero optimizer state, which is appropriate
        for clone and split children.
        """

        old_size = len(self)
        if not self._specs:
            raise RuntimeError("cannot mutate an empty, untyped registry")
        source = _as_index(source_indices, old_size, self.device)
        new_size = int(source.numel())
        if fresh_mask is None:
            fresh = torch.zeros(new_size, dtype=torch.bool, device=self.device)
        else:
            fresh = torch.as_tensor(fresh_mask, device=self.device, dtype=torch.bool).flatten()
            if fresh.numel() != new_size:
                raise ValueError(f"fresh_mask has {fresh.numel()} rows, expected {new_size}")
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(self._specs)
        if unknown:
            raise KeyError(f"overrides contain unregistered attributes: {sorted(unknown)}")

        prepared_values: dict[str, Tensor] = {}
        prepared_parameters: dict[str, nn.Parameter] = {}
        prepared_states: dict[str, dict[Any, Any]] = {}
        parameter_groups: dict[str, dict[str, Any]] = {}
        for group in self.optimizer.param_groups if self.optimizer is not None else []:
            for parameter in group["params"]:
                parameter_groups[str(id(parameter))] = group

        for name, spec in self._specs.items():
            old = self[name]
            value = overrides.get(name)
            if value is None:
                value = old.detach().index_select(0, source.to(old.device))
            else:
                value = torch.as_tensor(value, device=old.device, dtype=old.dtype)
            expected_shape = (new_size,) + spec.trailing_shape
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"override {name!r} has shape {tuple(value.shape)}, expected {expected_shape}")
            value = value.contiguous()
            prepared_values[name] = value
            if not spec.trainable:
                continue
            parameter = nn.Parameter(value, requires_grad=old.requires_grad)
            prepared_parameters[name] = parameter
            if self.optimizer is not None:
                group = parameter_groups.get(str(id(old)))
                if group is None:
                    raise RuntimeError(f"optimizer lost parameter group for {name!r}")
                old_state = self.optimizer.state.get(old, {})
                new_state: dict[Any, Any] = {}
                for key, state_value in old_state.items():
                    if isinstance(state_value, Tensor) and state_value.ndim > 0 and state_value.shape[0] == old_size:
                        transformed = state_value.index_select(0, source.to(state_value.device)).clone()
                        transformed[fresh.to(transformed.device)] = 0
                        new_state[key] = transformed
                    elif isinstance(state_value, Tensor):
                        new_state[key] = state_value.clone()
                    else:
                        new_state[key] = state_value
                prepared_states[name] = new_state

        old_parameters = {name: self[name] for name, spec in self._specs.items() if spec.trainable}
        # All validation and allocations happened above.  The following commit
        # consists solely of deterministic object-reference replacement.
        for name, spec in self._specs.items():
            if spec.trainable:
                self.trainable[name] = prepared_parameters[name]
            else:
                setattr(self, self._buffer_names[name], prepared_values[name])
        if self.optimizer is not None:
            for name, old in old_parameters.items():
                new = self[name]
                for group in self.optimizer.param_groups:
                    group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
                self.optimizer.state.pop(old, None)
                if prepared_states[name]:
                    self.optimizer.state[new] = prepared_states[name]
        return {name: self[name] for name in self._specs}

    def clone(
        self,
        selection: Tensor | Sequence[int],
        overrides: Mapping[str, Tensor] | None = None,
    ) -> int:
        indices = _as_index(selection, len(self), self.device)
        if not indices.numel():
            return 0
        old = torch.arange(len(self), device=self.device)
        source = torch.cat((old, indices))
        fresh = torch.cat(
            (torch.zeros_like(old, dtype=torch.bool), torch.ones_like(indices, dtype=torch.bool))
        )
        expanded: dict[str, Tensor] = {}
        for name, value in (overrides or {}).items():
            value = torch.as_tensor(value, device=self[name].device, dtype=self[name].dtype)
            if value.shape[0] == indices.numel():
                value = torch.cat((self[name].detach(), value), dim=0)
            expanded[name] = value
        self.mutate(source, overrides=expanded, fresh_mask=fresh)
        return int(indices.numel())

    def prune(self, selection: Tensor | Sequence[int]) -> int:
        indices = _as_index(selection, len(self), self.device)
        if not indices.numel():
            return 0
        keep = torch.ones(len(self), dtype=torch.bool, device=self.device)
        keep[indices] = False
        source = keep.nonzero(as_tuple=False).flatten()
        self.mutate(source)
        return int(indices.numel())

    def split(
        self,
        selection: Tensor | Sequence[int],
        children: int = 2,
        overrides: Mapping[str, Tensor] | None = None,
    ) -> int:
        if children < 1:
            raise ValueError("children must be positive")
        indices = _as_index(selection, len(self), self.device)
        if not indices.numel():
            return 0
        keep = torch.ones(len(self), dtype=torch.bool, device=self.device)
        keep[indices] = False
        survivors = keep.nonzero(as_tuple=False).flatten()
        child_source = indices.repeat_interleave(children)
        source = torch.cat((survivors, child_source))
        fresh = torch.cat(
            (
                torch.zeros_like(survivors, dtype=torch.bool),
                torch.ones_like(child_source, dtype=torch.bool),
            )
        )
        expanded: dict[str, Tensor] = {}
        for name, value in (overrides or {}).items():
            value = torch.as_tensor(value, device=self[name].device, dtype=self[name].dtype)
            child_shape = (indices.numel() * children,) + self._specs[name].trailing_shape
            if tuple(value.shape) == (indices.numel(), children) + self._specs[name].trailing_shape:
                value = value.flatten(0, 1)
            if tuple(value.shape) == child_shape:
                value = torch.cat((self[name].detach().index_select(0, survivors), value), dim=0)
            expanded[name] = value
        self.mutate(source, overrides=expanded, fresh_mask=fresh)
        return int(child_source.numel())

    def capture(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "specs": [asdict(spec) for spec in self._specs.values()],
            "tensors": {name: value.detach().clone() for name, value in self.named_attributes()},
        }

    def restore(self, snapshot: Mapping[str, Any], device: str | torch.device | None = None) -> None:
        if int(snapshot.get("format_version", 1)) > self.FORMAT_VERSION:
            raise ValueError("checkpoint was produced by a newer attribute registry")
        self.optimizer = None
        self.trainable = nn.ParameterDict()
        for name in list(self._buffer_names):
            delattr(self, self._buffer_names[name])
        self._buffer_names.clear()
        self._specs.clear()
        target_device = torch.device(device) if device is not None else None
        tensors = snapshot["tensors"]
        for serialized in snapshot["specs"]:
            data = dict(serialized)
            data["trailing_shape"] = tuple(data["trailing_shape"])
            spec = AttributeSpec(**data)
            value = tensors[spec.name]
            if target_device is not None:
                value = value.to(target_device)
            self.register(
                spec.name,
                value,
                trainable=spec.trainable,
                optimizer_group=spec.optimizer_group,
                role=spec.role,
                persistent=spec.persistent,
                ply_prefix=spec.ply_prefix,
                storage_order=spec.storage_order,
            )

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.capture(), path)

    def load_checkpoint(self, path: str | Path, device: str | torch.device | None = None) -> None:
        self.restore(torch.load(path, map_location=device or "cpu", weights_only=True), device)

    @staticmethod
    def _property_names(spec: AttributeSpec) -> list[str]:
        count = math.prod(spec.trailing_shape) if spec.trailing_shape else 1
        if spec.name == "xyz" and count == 3:
            return ["x", "y", "z"]
        if spec.name == "features_dc":
            return [f"f_dc_{index}" for index in range(count)]
        if spec.name == "features_rest":
            return [f"f_rest_{index}" for index in range(count)]
        if spec.name == "opacity" and count == 1:
            return ["opacity"]
        if spec.name == "scaling":
            return [f"scale_{index}" for index in range(count)]
        if spec.name == "rotation":
            return [f"rot_{index}" for index in range(count)]
        prefix = spec.ply_prefix or spec.name
        return [prefix] if count == 1 else [f"{prefix}_{index}" for index in range(count)]

    @staticmethod
    def _flatten(value: Tensor, spec: AttributeSpec) -> np.ndarray:
        value = value.detach().cpu()
        if spec.storage_order == "channel_sh" and value.ndim == 3:
            value = value.transpose(1, 2)
        return value.reshape(value.shape[0], -1).numpy()

    @staticmethod
    def _torch_dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    @staticmethod
    def _numpy_storage(dtype: np.dtype) -> tuple[str, np.dtype]:
        if np.issubdtype(dtype, np.floating):
            return ("double", np.dtype("<f8")) if dtype.itemsize > 4 else ("float", np.dtype("<f4"))
        if np.issubdtype(dtype, np.unsignedinteger):
            return ("uchar", np.dtype("u1")) if dtype.itemsize == 1 else ("uint", np.dtype("<u4"))
        if np.issubdtype(dtype, np.signedinteger):
            if dtype.itemsize <= 1:
                return "char", np.dtype("i1")
            if dtype.itemsize <= 2:
                return "short", np.dtype("<i2")
            if dtype.itemsize <= 4:
                return "int", np.dtype("<i4")
            return "double", np.dtype("<f8")
        if np.issubdtype(dtype, np.bool_):
            return "uchar", np.dtype("u1")
        raise TypeError(f"PLY cannot store dtype {dtype}")

    def save_ply(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        persistent = [(name, spec) for name, spec in self._specs.items() if spec.persistent]
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "attributes": [
                {
                    **asdict(spec),
                    "dtype": self._torch_dtype_name(self[name].dtype),
                    "properties": self._property_names(spec),
                }
                for name, spec in persistent
            ],
        }
        encoded = base64.urlsafe_b64encode(json.dumps(metadata, separators=(",", ":")).encode("utf8")).decode("ascii")
        columns: list[tuple[str, np.ndarray, str, np.dtype]] = []
        for name, spec in persistent:
            flat = self._flatten(self[name], spec)
            for index, property_name in enumerate(self._property_names(spec)):
                ply_type, storage_dtype = self._numpy_storage(flat.dtype)
                columns.append((property_name, flat[:, index], ply_type, storage_dtype))
        structured_dtype = np.dtype([(name, dtype) for name, _, _, dtype in columns])
        rows = np.empty(len(self), dtype=structured_dtype)
        for name, values, _, dtype in columns:
            rows[name] = values.astype(dtype, copy=False)
        with open(path, "wb") as stream:
            header = [
                "ply",
                "format binary_little_endian 1.0",
                f"comment semantic_gaussian_registry {encoded}",
                f"element vertex {len(self)}",
            ]
            header.extend(f"property {ply_type} {name}" for name, _, ply_type, _ in columns)
            header.append("end_header")
            stream.write(("\n".join(header) + "\n").encode("ascii"))
            rows.tofile(stream)

    @staticmethod
    def _read_ply(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
        type_map = {
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        with open(path, "rb") as stream:
            if stream.readline().strip() != b"ply":
                raise ValueError("not a PLY file")
            fmt, count, properties, encoded = None, None, [], None
            in_vertex = False
            while True:
                raw = stream.readline()
                if not raw:
                    raise EOFError("unterminated PLY header")
                line = raw.decode("ascii").strip()
                fields = line.split()
                if fields[:1] == ["format"]:
                    fmt = fields[1]
                elif fields[:2] == ["comment", "semantic_gaussian_registry"]:
                    encoded = fields[2]
                elif fields[:2] == ["element", "vertex"]:
                    count, in_vertex = int(fields[2]), True
                elif fields[:1] == ["element"]:
                    in_vertex = False
                elif fields[:1] == ["property"] and in_vertex:
                    if fields[1] == "list":
                        raise ValueError("list properties are invalid for Gaussian vertices")
                    if fields[1] not in type_map:
                        raise ValueError(f"unsupported PLY property type {fields[1]}")
                    properties.append((fields[2], np.dtype(type_map[fields[1]])))
                elif line == "end_header":
                    break
            if count is None:
                raise ValueError("PLY has no vertex element")
            if fmt == "binary_little_endian":
                rows = np.fromfile(stream, dtype=np.dtype(properties), count=count)
                columns = {name: rows[name] for name, _ in properties}
            elif fmt == "ascii":
                rows = [stream.readline().decode("ascii").split() for _ in range(count)]
                matrix = np.asarray(rows, dtype=np.float64).reshape(count, len(properties))
                columns = {name: matrix[:, index].astype(dtype) for index, (name, dtype) in enumerate(properties)}
            else:
                raise ValueError(f"unsupported PLY format {fmt!r}")
        metadata = None
        if encoded:
            metadata = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf8"))
        return columns, metadata

    @staticmethod
    def _dtype_from_name(name: str) -> torch.dtype:
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown torch dtype {name!r}")
        return dtype

    def load_ply(self, path: str | Path, device: str | torch.device | None = None) -> None:
        columns, metadata = self._read_ply(path)
        target_device = torch.device(device) if device is not None else self.device
        if metadata is not None:
            snapshot_specs, tensors = [], {}
            for item in metadata["attributes"]:
                item = dict(item)
                dtype = self._dtype_from_name(item.pop("dtype"))
                properties = item.pop("properties")
                shape = tuple(item["trailing_shape"])
                array = np.stack([columns[name] for name in properties], axis=1)
                tensor = torch.as_tensor(array.copy(), dtype=dtype, device=target_device)
                if item.get("storage_order") == "channel_sh" and len(shape) == 2:
                    tensor = tensor.reshape(len(tensor), shape[1], shape[0]).transpose(1, 2).contiguous()
                else:
                    tensor = tensor.reshape((len(tensor),) + shape)
                item["trailing_shape"] = shape
                snapshot_specs.append(item)
                tensors[item["name"]] = tensor
            self.restore(
                {"format_version": metadata.get("format_version", 1), "specs": snapshot_specs, "tensors": tensors},
                target_device,
            )
            return

        if not self._specs:
            raise ValueError("legacy 3DGS PLY requires a predeclared attribute schema")
        count = len(next(iter(columns.values()))) if columns else 0
        tensors = {}
        for name, spec in self._specs.items():
            properties = self._property_names(spec)
            if all(property_name in columns for property_name in properties):
                array = np.stack([columns[property_name] for property_name in properties], axis=1)
                tensor = torch.as_tensor(array.copy(), device=target_device, dtype=self[name].dtype)
                if spec.storage_order == "channel_sh" and len(spec.trailing_shape) == 2:
                    tensor = tensor.reshape(count, spec.trailing_shape[1], spec.trailing_shape[0]).transpose(1, 2).contiguous()
                else:
                    tensor = tensor.reshape((count,) + spec.trailing_shape)
            else:
                tensor = torch.zeros((count,) + spec.trailing_shape, dtype=self[name].dtype, device=target_device)
            tensors[name] = tensor
        self.restore(
            {"format_version": self.FORMAT_VERSION, "specs": [asdict(spec) for spec in self._specs.values()], "tensors": tensors},
            target_device,
        )


__all__ = ["AttributeSpec", "GaussianAttributeRegistry"]
