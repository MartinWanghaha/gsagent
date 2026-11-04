"""Opt-in inpaint clone/split/prune; never used by the global seg Trainer.

Rows carry an immutable initializer ancestor. Only new patch ancestors can
produce descendants. Frozen rows are buffers, including after topology changes.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from source.paintmesh_joint_data import JointGaussians


def camera_extent(views):
    centers = torch.stack([view.camera_center.detach() for view in views])
    radius = (centers - centers.mean(0)).norm(dim=1).max().item() * 1.1
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("adaptive density requires nondegenerate camera extent")
    return radius


class AdaptiveJointGaussians(JointGaussians):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("frozen_rows", torch.as_tensor(np.flatnonzero(~self.editable), device=self.indices.device))
        self.register_buffer("gradient_sum", torch.zeros(len(self.indices), device=self.indices.device))
        self.register_buffer("gradient_count", torch.zeros_like(self.gradient_sum))
        self.register_buffer("max_radii", torch.zeros_like(self.gradient_sum))
        self._set_indices(self.indices)
        self.density_counts = dict(events=0, cloned=0, split_parents=0, split_children=0,
                                   pruned_new=0, pruned_old=0, budget_limited=0, observed=0)

    def _set_indices(self, indices):
        self.indices = indices
        self.retained_rows = torch.cat((self.frozen_rows, indices))

    def assembled(self, key):
        return torch.cat((getattr(self, key + "_base")[self.frozen_rows], getattr(self, key)))

    @torch.no_grad()
    def accumulate(self, package):
        # Rasterizer order is fixed rows followed by the live trainable rows.
        start = len(self.frozen_rows)
        gradient = package["viewspace_points"].grad
        if gradient is None:
            raise ValueError("PGSR rasterizer did not return screen-space gradients")
        visible = package["visibility_filter"][start:]
        norms = gradient[start:, :2].norm(dim=-1)
        if norms.shape != self.gradient_sum.shape or not torch.isfinite(norms).all():
            raise ValueError("invalid adaptive density gradient shape/value")
        self.gradient_sum[visible] += norms[visible]
        self.gradient_count[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], package["radii"][start:][visible].float())
        self.density_counts["observed"] += int(visible.sum())

    @torch.no_grad()
    def _replace(self, optimizer, keep, parents=None, overrides=None, reset_statistics=False):
        """Retain Adam moments for survivors; appended children start with zero moments."""
        parents = self.indices.new_empty(0) if parents is None else parents
        overrides = overrides or {}
        for key in self.parameter_names:
            old = getattr(self, key)
            group = next(g for g in optimizer.param_groups if g["name"] == key)
            if group["params"][0] is not old or len(group["params"]) != 1:
                raise ValueError("adaptive density optimizer ownership mismatch")
            child = overrides.get(key, old[parents])
            new = nn.Parameter(torch.cat((old[keep], child)))
            state = optimizer.state.pop(old, None)
            if state is not None:
                optimizer.state[new] = {
                    k: torch.cat((v[keep], torch.zeros_like(child)))
                    if isinstance(v, torch.Tensor) and v.shape == old.shape else v
                    for k, v in state.items()}
            group["params"] = [new]
            setattr(self, key, new)
        for key in ("gradient_sum", "gradient_count", "max_radii"):
            value = getattr(self, key)
            new = torch.cat((value[keep], value.new_zeros(len(parents))))
            setattr(self, key, torch.zeros_like(new) if reset_statistics else new)
        self._set_indices(torch.cat((self.indices[keep], self.indices[parents])))

    @torch.no_grad()
    def prune(self, optimizer, min_opacity=0.005):
        keep = self.opacity.sigmoid().squeeze(1) >= min_opacity
        removed = int((~keep).sum())
        if not removed:
            return 0
        if not keep.any():
            raise ValueError("opacity pruning would remove all trainable Gaussians")
        new = torch.as_tensor(self.new_mask, device=self.indices.device)[self.indices]
        self.density_counts["pruned_new"] += int((~keep & new).sum())
        self.density_counts["pruned_old"] += int((~keep & ~new).sum())
        self._replace(optimizer, keep)
        return removed

    @torch.no_grad()
    def densify(self, optimizer, cfg, extent):
        average = self.gradient_sum / self.gradient_count.clamp_min(1)
        is_new = torch.as_tensor(self.new_mask, device=self.indices.device)[self.indices]
        candidates = torch.nonzero(is_new & (self.gradient_count > 0) & (average >= cfg["grad_threshold"])).flatten()
        # Both one clone and a two-child split increase the population by one.
        budget = max(0, cfg["max_new_points"] - int(is_new.sum()))
        if len(candidates) > budget:
            self.density_counts["budget_limited"] += len(candidates) - budget
            candidates = candidates[torch.argsort(average[candidates], descending=True, stable=True)[:budget]]
        small = self.scaling[candidates].exp().max(dim=1).values <= cfg["percent_dense"] * extent
        clone, split = candidates[small], candidates[~small]
        parents = torch.cat((clone, split.repeat(2)))
        child_xyz, child_scaling = self.xyz[parents].clone(), self.scaling[parents].clone()
        if len(split):
            # Same two-child sampling/scaling rule as the original 3DGS ADC.
            q = torch.nn.functional.normalize(self.rotation[split], dim=-1).repeat(2, 1)
            w, x, y, z = q.unbind(1)
            rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                                    2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                                    2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1).reshape(-1, 3, 3)
            samples = torch.normal(torch.zeros_like(self.xyz[split].repeat(2, 1)),
                                   self.scaling[split].exp().repeat(2, 1))
            child_xyz[len(clone):] += torch.bmm(rotation, samples.unsqueeze(-1)).squeeze(-1)
            child_scaling[len(clone):] -= np.log(0.8 * 2)
        keep = torch.ones(len(self.indices), dtype=torch.bool, device=self.indices.device)
        keep[split] = False
        self.density_counts["events"] += 1
        self.density_counts["cloned"] += len(clone)
        self.density_counts["split_parents"] += len(split)
        self.density_counts["split_children"] += 2 * len(split)
        if len(parents):
            self._replace(optimizer, keep, parents, dict(xyz=child_xyz, scaling=child_scaling), True)
        else:
            for key in ("gradient_sum", "gradient_count", "max_radii"):
                getattr(self, key).zero_()
        return dict(self.density_counts)

    def local_state(self):
        return dict(parameters=super().local_state(), density_counts=dict(self.density_counts),
                    statistics={k: getattr(self, k).detach().clone()
                                for k in ("gradient_sum", "gradient_count", "max_radii")})

    @torch.no_grad()
    def restore_local(self, state):
        if set(state) != {"parameters", "statistics", "density_counts"}:
            raise ValueError("invalid adaptive density checkpoint")
        values = state["parameters"]
        if set(values) != {*self.parameter_names, "indices"}:
            raise ValueError("invalid adaptive density parameter fields")
        indices = values["indices"]
        if (indices.dtype != torch.int64 or indices.ndim != 1 or not len(indices)
                or (indices < 0).any() or (indices >= len(self.editable)).any()
                or not self.editable[indices.cpu().numpy()].all()):
            raise ValueError("invalid adaptive density ancestor ownership")
        old = indices[~torch.as_tensor(self.new_mask)[indices.cpu()]]
        if len(old.unique()) != len(old):
            raise ValueError("adaptive density duplicated an old row")
        for key in self.parameter_names:
            value, parameter = values[key], getattr(self, key)
            if (value.shape != (len(indices), *parameter.shape[1:]) or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError("invalid adaptive density checkpoint parameters")
            setattr(self, key, nn.Parameter(value.to(parameter).clone()))
        if set(state["statistics"]) != {"gradient_sum", "gradient_count", "max_radii"}:
            raise ValueError("invalid adaptive density statistics")
        for key, value in state["statistics"].items():
            if value.shape != (len(indices),) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError("invalid adaptive density statistics")
            setattr(self, key, value.to(self.gradient_sum).clone())
        counts = state["density_counts"]
        if set(counts) != set(self.density_counts) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("invalid adaptive density counters")
        self.density_counts = dict(counts)
        self._set_indices(indices.to(self.indices.device).clone())

    @torch.no_grad()
    def save(self, path):
        from edgs_inpaint_io import atomic_write, check_preserved
        from plyfile import PlyData, PlyElement
        rows = self.retained_rows.cpu().numpy()
        vertex = self.ply["vertex"].data[rows].copy()
        selected = slice(len(self.frozen_rows), None)
        for key, fields in self.groups.items():
            values = getattr(self, key).cpu().numpy()
            for i, field in enumerate(fields):
                vertex[field][selected] = values[:, i]
        features = self.features.cpu().numpy()
        for i in range(3):
            vertex[f"f_dc_{i}"][selected] = features[:, 0, i]
        rest = features[:, 1:].transpose(0, 2, 1).reshape(len(features), -1)
        for i in range(rest.shape[1]):
            vertex[f"f_rest_{i}"][selected] = rest[:, i]
        vertex["opacity"][selected] = self.opacity[:, 0].cpu().numpy()
        atomic_write(Path(path), lambda f: PlyData([PlyElement.describe(vertex, "vertex")]).write(f))
        check_preserved(self.ply_path, path, self.new_mask, retained_rows=rows,
                        freeze_background=self.freeze_background, trainable_mask=self.editable, new_descendants=True)
