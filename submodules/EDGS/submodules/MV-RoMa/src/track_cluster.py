import torch
from typing import Tuple, List, Optional, Dict, Any
import math


@torch.inference_mode()
def extract_tracks_match(
    coords_q: torch.Tensor,   # [H,W,2]
    coords_refs: torch.Tensor,    # [R,H,W,2]
    covis, # [R,H,W] bool
    N: int,
    covisibility_threshold: float = 0.3,
    downsample_stride: int = 8,
    kmeans_iters: int = 12,
    kmeans_seed: int = 0,
    cluster_mode: str = "global",
    device=None,
) -> torch.Tensor:
    """
    Same semantics as extract_tracks_ufm, but clustering runs fully on GPU.

    Returns:
        tracks: [R, N_eff, 4] float32 on GPU.
        For each ref r and sample t: [x_q, y_q, x_r, y_r].
        For refs not visible in a track, [x_r, y_r] stay at -1000.
    """
    if device is None:
        device = coords_q.device

    if cluster_mode not in ("per-group", "global"):
        raise ValueError(f"Unsupported cluster_mode: {cluster_mode}")

    H, W = coords_q.shape[:2]
    R = coords_refs.shape[0]

    # Visibility and grouping
    vis = covis > covisibility_threshold                                   # [R,H,W] bool

    if not vis.any():
        return torch.empty((R, 0, 4), device=device, dtype=torch.float32)

    codes = torch.zeros((H, W), dtype=torch.int32, device=device)
    for i in range(R):
        codes |= (vis[i].to(torch.int32) << i)

    present_codes = torch.unique(codes)
    present_codes = present_codes[present_codes != 0]

    # Build list of (code, ref_idxs, count), sorted by richness then count
    pattern_list: List[Tuple[int, Tuple[int, ...], int]] = []
    for code in present_codes.tolist():
        mask = (codes == code)
        cnt = int(mask.sum().item())
        if cnt > 0:
            ref_idxs = tuple(i for i in range(R) if (code >> i) & 1)
            pattern_list.append((code, ref_idxs, cnt))
    pattern_list.sort(key=lambda x: (-len(x[1]), -x[2], x[0]))

    if not pattern_list:
        return torch.empty((R, 0, 4), device=device, dtype=torch.float32)

    # Capacity caps and proportional allocation
    caps = [max(1, (cnt + downsample_stride - 1) // downsample_stride) for (_, _, cnt) in pattern_list]
    total_cap = int(sum(caps))
    if total_cap == 0:
        return torch.empty((R, 0, 4), device=device, dtype=torch.float32)

    N_eff = min(int(N), total_cap)
    raw_fracs = [N_eff * (c / total_cap) for c in caps]
    ks = [min(caps[i], int(raw_fracs[i])) for i in range(len(caps))]
    assigned = sum(ks)
    frac_parts = sorted(
        [(raw_fracs[i] - int(raw_fracs[i]), i) for i in range(len(caps))],
        key=lambda x: -x[0]
    )
    rem = N_eff - assigned
    for _, i in frac_parts:
        if rem == 0:
            break
        if ks[i] < caps[i]:
            ks[i] += 1
            rem -= 1
    if rem > 0:
        for i in range(len(caps)):
            if rem == 0:
                break
            free = caps[i] - ks[i]
            if free > 0:
                give = min(free, rem)
                ks[i] += give
                rem -= give

    stride = max(1, int(downsample_stride))

    # Prepass for effective k after downsampling
    k_use_list: List[int] = []
    samples_idx: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for (code, _, _), k_i in zip(pattern_list, ks):
        mask = (codes == code)
        yi, xi = mask.nonzero(as_tuple=True)
        yi = yi[::stride]
        xi = xi[::stride]
        M = yi.numel()
        samples_idx.append((yi, xi))
        k_use_list.append(min(k_i, int(M)))

    N_eff = int(sum(k_use_list))
    if N_eff == 0:
        return torch.empty((R, 0, 4), device=device, dtype=torch.float32)

    out = torch.full((R, N_eff, 4), float("-1000."), device=device, dtype=torch.float32)
    t = 0

    if cluster_mode == "per-group":
        for ((_, ref_idxs, _), k_i, (yi, xi)) in zip(pattern_list, k_use_list, samples_idx):
            k_use = int(k_i)
            if k_use <= 0:
                continue

            feats = [coords_q[yi, xi]]
            for r in ref_idxs:
                feats.append(coords_refs[r, yi, xi])
            group = torch.cat(feats, dim=1).to(torch.float32)

            centers = _torch_kmeans_gpu(group, k=k_use, iters=kmeans_iters, seed=kmeans_seed)
            k_actual = int(centers.shape[0])
            if k_actual <= 0:
                continue

            out[:, t:t + k_actual, 0] = centers[:, 0].unsqueeze(0)
            out[:, t:t + k_actual, 1] = centers[:, 1].unsqueeze(0)
            for r in ref_idxs:
                pos = ref_idxs.index(r)
                out[r, t:t + k_actual, 2] = centers[:, 2 + 2 * pos + 0]
                out[r, t:t + k_actual, 3] = centers[:, 2 + 2 * pos + 1]
            t += k_actual
    else:
        # Global mode: one k-means over a fixed-width feature matrix [x_q, y_q, x_r0, y_r0, ...]
        D = 2 + 2 * R
        missing_sentinel = -100000.0
        M_total = int(sum(int(yi.numel()) for (k_i, (yi, _)) in zip(k_use_list, samples_idx) if int(k_i) > 0))
        if M_total <= 0:
            return torch.empty((R, 0, 4), device=device, dtype=torch.float32)

        x_global = torch.full((M_total, D), missing_sentinel, device=device, dtype=torch.float32)
        row = 0
        for ((_, ref_idxs, _), k_i, (yi, xi)) in zip(pattern_list, k_use_list, samples_idx):
            if int(k_i) <= 0:
                continue
            m = int(yi.numel())
            if m <= 0:
                continue
            span = slice(row, row + m)
            x_global[span, 0:2] = coords_q[yi, xi]
            for r in ref_idxs:
                start = 2 + 2 * r
                x_global[span, start:start + 2] = coords_refs[r, yi, xi]
            row += m
        x_global = x_global[:row]

        centers = _torch_kmeans_gpu(x_global, k=N_eff, iters=kmeans_iters, seed=kmeans_seed)
        k_actual = int(centers.shape[0])
        out = torch.full((R, k_actual, 4), float("-1000."), device=device, dtype=torch.float32)
        if k_actual > 0:
            out[:, :, 0] = centers[:, 0].unsqueeze(0)
            out[:, :, 1] = centers[:, 1].unsqueeze(0)
            for r in range(R):
                start = 2 + 2 * r
                xr = centers[:, start + 0]
                yr = centers[:, start + 1]
                valid = (xr != missing_sentinel) & (yr != missing_sentinel)
                if valid.any():
                    out[r, valid, 2] = xr[valid]
                    out[r, valid, 3] = yr[valid]
        t = k_actual

    return out[:, :t, :]


def _snap_centroids_to_assigned_samples(
    x: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Vectorized medoid snap with deterministic lowest-index tie breaking."""

    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("samples and centers must have compatible [N,D] shapes")
    if centers.shape[0] == 0 or x.shape[0] == 0:
        return centers.clone()

    distances = (
        (x * x).sum(dim=1, keepdim=True)
        + (centers * centers).sum(dim=1).unsqueeze(0)
        - 2.0 * (x @ centers.t())
    )
    labels = distances.argmin(dim=1)
    point_indices = torch.arange(x.shape[0], device=x.device)
    assigned_distance = distances[point_indices, labels]
    minimum_distance = torch.full(
        (centers.shape[0],),
        float("inf"),
        device=x.device,
        dtype=assigned_distance.dtype,
    )
    minimum_distance.scatter_reduce_(
        0,
        labels,
        assigned_distance,
        reduce="amin",
        include_self=True,
    )
    candidate_indices = torch.where(
        assigned_distance == minimum_distance[labels],
        point_indices,
        torch.full_like(point_indices, x.shape[0]),
    )
    closest_indices = torch.full(
        (centers.shape[0],),
        x.shape[0],
        device=x.device,
        dtype=torch.long,
    )
    closest_indices.scatter_reduce_(
        0,
        labels,
        candidate_indices,
        reduce="amin",
        include_self=True,
    )
    snapped = centers.clone()
    populated = closest_indices < x.shape[0]
    snapped[populated] = x[closest_indices[populated]]
    return snapped


def _torch_kmeans_gpu(x: torch.Tensor, k: int, iters: int = 12, seed: int = 0) -> torch.Tensor:
    """
    Lloyd's k-means entirely on GPU.
    x: [M, D] float32 CUDA tensor
    returns: centers [k, D] float32 CUDA tensor
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 2
    M, D = x.shape
    if k <= 0 or M == 0:
        return x.new_empty((0, D))
    if k == 1:
        return x.mean(dim=0, keepdim=True)

    g = torch.Generator(device=x.device).manual_seed(seed)
    if k <= M:
        perm = torch.randperm(M, generator=g, device=x.device)
        c = x[perm[:k]].clone()
    else:
        idx = torch.randint(low=0, high=M, size=(k,), generator=g, device=x.device)
        c = x[idx].clone()

    x_sq = (x * x).sum(dim=1, keepdim=True)  # [M,1]

    for _ in range(max(1, iters)):
        c_sq = (c * c).sum(dim=1).unsqueeze(0)               # [1,k]
        dist = x_sq + c_sq - 2.0 * (x @ c.t())               # [M,k]
        labels = dist.argmin(dim=1)                          # [M]

        new_c = torch.zeros_like(c)
        assign = labels.unsqueeze(1).expand(-1, D)           # [M,D]
        new_c.scatter_add_(0, assign, x)

        counts = torch.bincount(labels, minlength=k)         # [k]
        has_pts = counts > 0
        if has_pts.any():
            new_c[has_pts] = new_c[has_pts] / counts[has_pts].unsqueeze(1).to(x.dtype)

        if (~has_pts).any():
            empty_idx = (~has_pts).nonzero(as_tuple=False).squeeze(1)
            n_empty = int(empty_idx.numel())
            point_dists = dist[torch.arange(M, device=x.device), labels]
            if M >= n_empty:
                repl = torch.topk(point_dists, k=n_empty, largest=True, sorted=False).indices
            else:
                head = torch.topk(point_dists, k=M, largest=True, sorted=False).indices
                tail = torch.randint(low=0, high=M, size=(n_empty - M,), generator=g, device=x.device)
                repl = torch.cat((head, tail), dim=0)
            new_c[empty_idx] = x[repl]

        if torch.allclose(new_c, c, rtol=0.0, atol=1e-4):
            c = new_c
            break
        c = new_c

    # The upstream Python loop synchronized CUDA once per cluster. This
    # vectorized equivalent preserves the same assigned-sample semantics.
    return _snap_centroids_to_assigned_samples(x, c)
