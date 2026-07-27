"""Per-Gaussian statistics from the rasterizer: conflict, residual, variance.

Two complementary sources feed the Semantic Prior Field density control:

1. **CUDA stats channel** (``diff_gaussian_rasterization_spf``): the semantic
   backward accumulates, per Gaussian, the unsigned gradient mass
   ``sum_p w_p ||dL/dE(p)||`` and the visibility mass ``sum_p w_p`` at zero
   extra cost. Together with the signed embedding gradient this yields a
   *conflict score*: boundary-straddling Gaussians receive opposing
   per-pixel gradients that cancel in the signed sum but not in the
   unsigned one.

2. **Transposed-render scatter** (pure PyTorch, any rasterizer): the
   backward of rendering w.r.t. ``colors_precomp`` is exactly the operator
   "scatter a per-pixel map onto Gaussians with weights T*alpha". Rendering
   a dummy color and backpropagating ``(render * error_map).sum()`` gives a
   contribution-weighted error attribution that is strictly better than
   argmax (gidx) attribution.

The accumulator collects both across iterations and exposes normalized
scores; it follows the same invalidate-on-topology-change discipline as the
prior field.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class SemanticStatsAccumulator:
    """Running per-Gaussian statistics collected during normal training.

    All buffers are (N,) and are reset whenever the Gaussian count changes.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.abs_grad = None          # sum over iters of sum_p w ||dL/dE||
        self.signed_grad_norm = None  # sum over iters of ||sum_p w dL/dE||
        self.contribution = None      # sum over iters of sum_p w
        self.residual = None          # sum over iters of sum_p w |photometric residual|
        self.residual_sq = None       # sum over iters of sum_p w residual^2
        self.updates = 0

    def _ensure(self, n_gaussians: int) -> None:
        if self.abs_grad is None or self.abs_grad.shape[0] != n_gaussians:
            self.reset(n_gaussians)

    def reset(self, n_gaussians: Optional[int] = None) -> None:
        if n_gaussians is None:
            self.abs_grad = None
            self.signed_grad_norm = None
            self.contribution = None
            self.residual = None
            self.residual_sq = None
        else:
            zeros = lambda: torch.zeros(n_gaussians, device=self.device)  # noqa: E731
            self.abs_grad = zeros()
            self.signed_grad_norm = zeros()
            self.contribution = zeros()
            self.residual = zeros()
            self.residual_sq = zeros()
        self.updates = 0

    @torch.no_grad()
    def update_from_sink(self, sink: Dict[str, torch.Tensor]) -> None:
        """Consume one iteration of CUDA stats (spf rasterizer backward)."""
        abs_grad = sink.get("semantic_abs_grad")
        if abs_grad is None:
            return
        self._ensure(abs_grad.shape[0])
        self.abs_grad += abs_grad
        self.contribution += sink["semantic_contribution"]
        semantic_grad = sink.get("semantic_grad")
        if semantic_grad is not None:
            self.signed_grad_norm += semantic_grad.norm(dim=-1)
        self.updates += 1

    @torch.no_grad()
    def update_residual(self, residual: torch.Tensor, weight: torch.Tensor) -> None:
        """Accumulate contribution-weighted photometric residual scatter."""
        self._ensure(residual.shape[0])
        self.residual += residual
        self.residual_sq += residual ** 2 / weight.clamp_min(1e-8)
        # residual_sq accumulates (sum_p w r)^2 / sum_p w per view, an
        # inter-view dispersion proxy once combined in residual_variance().

    @property
    def ready(self) -> bool:
        return self.updates > 0 and self.abs_grad is not None

    @torch.no_grad()
    def conflict_score(self) -> Optional[torch.Tensor]:
        """(N,) boundary/conflict score in [0, inf).

        ``(unsigned - signed) / contribution``: zero for Gaussians whose
        semantic gradients agree across their footprint, large for Gaussians
        pulled toward different classes by different pixels/views.
        """
        if not self.ready:
            return None
        conflict = (self.abs_grad - self.signed_grad_norm).clamp_min(0.0)
        return conflict / self.contribution.clamp_min(1e-8)

    @torch.no_grad()
    def error_mass_score(self) -> Optional[torch.Tensor]:
        """(N,) plain normalized semantic gradient mass (error magnitude)."""
        if not self.ready:
            return None
        return self.abs_grad / self.contribution.clamp_min(1e-8)

    @torch.no_grad()
    def mean_residual(self) -> Optional[torch.Tensor]:
        if self.residual is None or self.contribution is None:
            return None
        return self.residual / self.contribution.clamp_min(1e-8)


@torch.no_grad()
def _detach_render_inputs():
    pass


def transposed_error_scatter(
    render_func,
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    error_map: torch.Tensor,
) -> tuple:
    """Contribution-weighted scatter of a per-pixel map onto Gaussians.

    Runs one lightweight 3-channel render with a dummy ``colors_precomp``
    and backpropagates only into the dummy: its gradient is exactly
    ``sum_p T(p) alpha(p) error(p)`` per Gaussian. Also returns the
    visibility mass obtained the same way with a unit map.

    Args:
        render_func: the module-level render (radegs/ours signature).
        error_map: (H, W) non-negative per-pixel quantity.

    Returns:
        (scatter (N,), weight (N,)): error mass and visibility mass.
    """
    n_gaussians = gaussians.get_xyz.shape[0]
    dummy = torch.zeros(
        n_gaussians, 3, device=error_map.device, requires_grad=True
    )
    # Channel 0 carries the error scatter, channel 1 the visibility mass.
    render_pkg = render_func(
        viewpoint_camera=viewpoint_cam,
        pc=gaussians,
        pipe=pipe,
        bg_color=background,
        colors_precomp=dummy + torch.tensor(
            [0.0, 1.0, 0.0], device=error_map.device
        ),
        require_coord=False,
        require_depth=False,
    )
    rendered = render_pkg["render"]  # (3, H, W)
    target = torch.stack(
        [error_map.detach(), torch.ones_like(error_map)], dim=0
    )  # (2, H, W)
    proxy_loss = (rendered[:2] * target).sum()
    (grad,) = torch.autograd.grad(proxy_loss, dummy, retain_graph=False)
    return grad[:, 0].detach(), grad[:, 1].detach()
