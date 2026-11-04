"""Inpaint360GS-style train-all, restore-background and local writeback.

Only the joint inpaint entry point uses this module. The seed is the unchanged
initializer's NEW cloud, so the ablation does not introduce a new initialization.
"""
import numpy as np
import torch


@torch.no_grad()
def select_writeback(xyz, full_projection, mask, seed_xyz, config):
    import cv2
    from scipy.spatial import cKDTree
    mask = np.asarray(mask, dtype=bool)
    seed = np.asarray(seed_xyz, dtype=np.float32)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("first-view writeback mask must be a nonempty 2D mask")
    if (seed.ndim != 2 or seed.shape[1] != 3 or not len(seed)
            or not np.isfinite(seed).all() or xyz.ndim != 2 or xyz.shape[1] != 3
            or not torch.isfinite(xyz).all() or full_projection.shape != (4, 4)):
        raise ValueError("invalid writeback seed/points/projection")
    original_area = int(mask.sum())
    target_area = int(original_area * config["mask_area_factor"])
    for kernel_size in range(3, 101, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        if expanded.sum() >= target_area:
            break
    height, width = mask.shape
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1) @ full_projection.to(xyz)
    projected = homogeneous[:, :2] * (1 / (homogeneous[:, 3:] + 1e-8))
    pixels = torch.round((projected + 1) * .5 * projected.new_tensor([width - 1, height - 1])).long()
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (homogeneous[:, 2] > 0))
    candidates = torch.nonzero(inside).flatten()
    pixel = pixels[candidates].cpu().numpy()
    candidates = candidates[torch.as_tensor(expanded[pixel[:, 1], pixel[:, 0]], device=xyz.device)]
    threshold = float(np.std(seed, axis=0).mean() * config["seed_std_factor"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("writeback seed cloud must have nonzero spatial extent")
    distances = cKDTree(seed).query(xyz[candidates].detach().cpu().numpy(), workers=4)[0]
    selected = np.zeros(len(xyz), dtype=bool)
    selected[candidates.cpu().numpy()[distances < threshold]] = True
    return selected, dict(mask_area=original_area, expanded_area=int(expanded.sum()),
        mask_kernel_size=kernel_size, mask_area_target=target_area,
        seed_count=len(seed), seed_distance_threshold=threshold,
        projection_candidates=len(candidates), selected_count=int(selected.sum()))


@torch.no_grad()
def restore_background(model, selected):
    """Restore all unselected OLD parameters, retain all surviving NEW rows."""
    rows = model.retained_rows.cpu().numpy()
    selected = np.asarray(selected)
    if selected.dtype != np.bool_ or selected.shape != rows.shape:
        raise ValueError("writeback selection must index the live trained cloud")
    if not np.array_equal(rows[~model.new_mask[rows]], np.flatnonzero(~model.new_mask)):
        raise ValueError("cannot restore background: an original old row is missing")
    permitted = model.new_mask.copy()
    permitted[rows[selected]] = True
    restore = ~torch.as_tensor(permitted, device=model.indices.device)[model.indices]
    changed_old = torch.zeros(len(model.indices), dtype=torch.bool, device=model.indices.device)
    old = ~torch.as_tensor(model.new_mask, device=model.indices.device)[model.indices]
    for key in model.parameter_names:
        parameter = getattr(model, key)
        original = getattr(model, key + "_base")[model.indices]
        changed_old |= (parameter != original).reshape(len(parameter), -1).any(1) & old
        parameter[restore] = original[restore]
    return permitted, dict(selected_old=int((permitted & ~model.new_mask).sum()),
        restored_old=int((~permitted).sum()),
        changed_old_before_restore=int(changed_old.sum()),
        changed_old_kept=int((changed_old & ~restore).sum()),
        changed_old_restored=int((changed_old & restore).sum()),
        surviving_new=int(model.new_mask[rows].sum()))
