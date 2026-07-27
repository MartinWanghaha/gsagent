"""Validation experiments for the SPF rasterizer stats channel.

Run directly on a CUDA machine:

    python tests/exp_spf_stats_validation.py

Verifies, on a synthetic two-instance scene:

  E1  forward parity: the SPF backend reproduces the ours-semantic backend
      bit-for-bit (color / semantic / depth / alpha);
  E2  backward parity: identical embedding gradients;
  E3  contribution identity: with dL/dE = one-hot(c) everywhere,
      stat_contribution == grad_semantic[:, c] exactly;
  E4  aligned-gradient identity: with a spatially constant unit gradient,
      unsigned mass == signed mass (conflict == 0);
  E5  conflict localization: with an antisymmetric gradient across a
      vertical boundary, conflict concentrates on boundary-straddling
      Gaussians;
  E6  overhead: SPF backward wall time vs ours backward.
"""

import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
# ours-semantic ships an in-source build; the SPF backend must come from the
# installed package (pip install submodules/diff-gaussian-rasterization-spf).
sys.path[:0] = [
    str(PROJECT / "submodules" / "diff-gaussian-rasterization_ours-semantic"),
]

import diff_gaussian_rasterization_gw_ours_semantic as ours_ext  # noqa: E402
import diff_gaussian_rasterization_spf as spf_ext  # noqa: E402


def projection_matrix(znear, zfar, fovx, fovy, device):
    tan_half_fovx = math.tan(fovx / 2)
    tan_half_fovy = math.tan(fovy / 2)
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right
    P = torch.zeros(4, 4, device=device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def make_scene(n=3000, seed=0, device="cuda"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    xyz = torch.rand(n, 3, generator=generator) * 2.0 - 1.0
    xyz[:, 2] = xyz[:, 2] * 0.5 + 3.0  # z in [2.5, 3.5]
    xyz = xyz.to(device)
    scales = torch.full((n, 3), 0.03, device=device)
    rotations = torch.zeros(n, 4, device=device)
    rotations[:, 0] = 1.0
    opacities = torch.full((n, 1), 0.8, device=device)
    colors = torch.rand(n, 3, generator=generator).to(device)
    # Two instances split at x = 0, embeddings on channels 0 / 1
    semantic = torch.zeros(n, 16, device=device)
    left = xyz[:, 0] < 0
    semantic[left, 0] = 1.0
    semantic[~left, 1] = 1.0
    return xyz, scales, rotations, opacities, colors, semantic


def make_settings(ext, H=160, W=160, device="cuda", sink=None):
    fov = math.radians(60.0)
    view = torch.eye(4, device=device)  # camera at origin, +z forward
    proj = projection_matrix(0.01, 100.0, fov, fov, device)
    full_proj = (view.t().unsqueeze(0).bmm(proj.t().unsqueeze(0))).squeeze(0)
    kwargs = dict(
        image_height=H,
        image_width=W,
        tanfovx=math.tan(fov / 2),
        tanfovy=math.tan(fov / 2),
        kernel_size=0.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view.t(),
        projmatrix=full_proj,
        sh_degree=0,
        sg_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        require_depth=True,
        debug=False,
    )
    if sink is not None:
        kwargs["stats_sink"] = sink
    return ext.GaussianRasterizationSettings(**kwargs)


def run_forward(ext, scene, sink=None, grad_map=None):
    xyz, scales, rotations, opacities, colors, semantic = scene
    n = xyz.shape[0]
    device = xyz.device
    semantic = semantic.clone().requires_grad_(True)
    settings = make_settings(ext, sink=sink, device=device)
    rasterizer = ext.GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(xyz, requires_grad=True)
    outputs = rasterizer(
        means3D=xyz,
        means2D=means2D,
        opacities=opacities,
        semantic_features=semantic,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
        sg_axis=torch.zeros(n, 0, 3, device=device),
        sg_sharpness=torch.zeros(n, 0, device=device),
        sg_color=torch.zeros(n, 0, 3, device=device),
    )
    color, semantic_map, radii, mdepth, alpha, normal = outputs
    grad_e = None
    if grad_map is not None:
        loss = (semantic_map * grad_map).sum()  # dL/dE == grad_map exactly
        loss.backward()
        grad_e = semantic.grad.detach().clone()
    return {
        "color": color.detach(), "semantic": semantic_map.detach(),
        "mdepth": mdepth.detach(), "alpha": alpha.detach(),
        "radii": radii, "grad_e": grad_e,
    }


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    scene = make_scene(device=device)
    H = W = 160
    results = {}

    # --- E1/E2: parity between backends -----------------------------------
    grad_map = torch.randn(16, H, W, device=device)
    ours_out = run_forward(ours_ext, scene, grad_map=grad_map)
    sink = {}
    spf_out = run_forward(spf_ext, scene, sink=sink, grad_map=grad_map)
    for key in ("color", "semantic", "mdepth", "alpha"):
        max_diff = (ours_out[key] - spf_out[key]).abs().max().item()
        results[f"E1 forward parity [{key}] max|diff|"] = max_diff
        assert max_diff == 0.0, f"forward mismatch in {key}: {max_diff}"
    # atomicAdd float summation order differs once the stats atomics join the
    # kernel, so the comparison is numerical, not bitwise.
    grad_diff = (ours_out["grad_e"] - spf_out["grad_e"]).abs().max().item()
    grad_scale = ours_out["grad_e"].abs().max().item()
    results["E2 backward parity [grad_e] max|diff|"] = grad_diff
    assert grad_diff <= 1e-4 * max(grad_scale, 1.0), f"backward mismatch: {grad_diff}"
    assert "semantic_abs_grad" in sink, "stats sink not populated"

    # --- E3: contribution identity ----------------------------------------
    onehot = torch.zeros(16, H, W, device=device)
    onehot[3] = 1.0
    sink3 = {}
    out3 = run_forward(spf_ext, scene, sink=sink3, grad_map=onehot)
    contribution = sink3["semantic_contribution"]
    identity_err = (contribution - out3["grad_e"][:, 3]).abs().max().item()
    results["E3 contribution identity max|diff|"] = identity_err
    assert identity_err < 1e-4, f"contribution identity broken: {identity_err}"

    # --- E4: aligned gradient => zero conflict ----------------------------
    direction = torch.zeros(16, device=device)
    direction[5] = 0.6
    direction[7] = 0.8  # unit norm
    aligned = direction.view(16, 1, 1).expand(16, H, W).contiguous()
    sink4 = {}
    out4 = run_forward(spf_ext, scene, sink=sink4, grad_map=aligned)
    signed_norm = out4["grad_e"].norm(dim=-1)
    conflict = (sink4["semantic_abs_grad"] - signed_norm).abs().max().item()
    results["E4 aligned-gradient conflict max"] = conflict
    assert conflict < 1e-3, f"aligned gradients must not conflict: {conflict}"

    # --- E5: conflict localizes at the boundary ---------------------------
    xs = torch.linspace(-1, 1, W, device=device)
    sign_map = torch.sign(xs).view(1, 1, W).expand(1, H, W)
    anti = torch.zeros(16, H, W, device=device)
    anti[5] = sign_map[0]
    sink5 = {}
    out5 = run_forward(spf_ext, scene, sink=sink5, grad_map=anti)
    signed_norm5 = out5["grad_e"].norm(dim=-1)
    conflict5 = (sink5["semantic_abs_grad"] - signed_norm5).clamp_min(0)
    conflict5 = conflict5 / sink5["semantic_contribution"].clamp_min(1e-8)
    xyz = scene[0]
    visible = out5["radii"] > 0
    top = conflict5[visible].topk(k=min(100, int(visible.sum())))
    top_x = xyz[visible][top.indices][:, 0].abs()
    rest_x = xyz[visible][:, 0].abs().median()
    results["E5 |x| of top-conflict Gaussians (median)"] = top_x.median().item()
    results["E5 |x| of all visible Gaussians (median)"] = rest_x.item()
    assert top_x.median() < 0.35 * rest_x, (
        "conflict should concentrate near the x=0 boundary: "
        f"{top_x.median().item()} vs {rest_x.item()}"
    )

    # --- E6: overhead ------------------------------------------------------
    def timed(ext, sink_factory, iters=20):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            run_forward(ext, scene, sink=sink_factory(), grad_map=grad_map)
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000.0

    ours_ms = timed(ours_ext, lambda: None)
    spf_ms = timed(spf_ext, lambda: {})
    results["E6 ours fw+bw (ms)"] = ours_ms
    results["E6 spf fw+bw (ms)"] = spf_ms
    results["E6 overhead (%)"] = (spf_ms / ours_ms - 1.0) * 100.0

    print("\n=== SPF rasterizer stats-channel validation ===")
    for name, value in results.items():
        print(f"  {name}: {value:.6g}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
