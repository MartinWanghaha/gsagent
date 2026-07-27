"""Gradient conflict protection for joint rendering and geometry objectives."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class PhotometricParetoGuard:
    """Project conflicting auxiliary gradients away from the RGB gradient.

    This is a two-task specialization of PCGrad.  It preserves the component of
    the auxiliary gradient orthogonal to the photometric objective instead of
    simply lowering every semantic/mesh loss with a hand-tuned scalar.
    """

    def __init__(self, enabled: bool = True, eps: float = 1e-12) -> None:
        self.enabled = enabled
        self.eps = eps
        self.last_cosine = 0.0
        self.last_projected = False

    @staticmethod
    def _grad(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> tuple[torch.Tensor | None, ...]:
        return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)

    def backward(
        self,
        photometric_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        protected_parameters: Iterable[torch.nn.Parameter],
    ) -> None:
        parameters = [parameter for parameter in protected_parameters if parameter.requires_grad]
        if not self.enabled or not parameters or not auxiliary_loss.requires_grad:
            (photometric_loss + auxiliary_loss).backward()
            return

        # One explicit photometric VJP plus one ordinary auxiliary backward is
        # sufficient. Keeping a second autograd.grad(auxiliary) followed by an
        # auxiliary backward would traverse the large raster graph three times.
        photo_grads = self._grad(photometric_loss, parameters)
        auxiliary_loss.backward()
        aux_grads = tuple(parameter.grad for parameter in parameters)

        dot = photometric_loss.new_zeros(())
        photo_norm = photometric_loss.new_zeros(())
        aux_norm = photometric_loss.new_zeros(())
        for photo, aux in zip(photo_grads, aux_grads):
            if photo is None or aux is None:
                continue
            dot = dot + (photo * aux).sum()
            photo_norm = photo_norm + photo.square().sum()
            aux_norm = aux_norm + aux.square().sum()

        denominator = (photo_norm.sqrt() * aux_norm.sqrt()).clamp_min(self.eps)
        self.last_cosine = float((dot / denominator).detach())
        self.last_projected = bool(dot.detach() < 0)

        # Auxiliary backward above also populated parameters exclusive to the
        # semantic/mesh heads. Replace only protected Gaussian gradients.
        coefficient = (dot / photo_norm.clamp_min(self.eps)).detach() if self.last_projected else dot.new_zeros(())
        for parameter, photo, aux in zip(parameters, photo_grads, aux_grads):
            if photo is None and aux is None:
                continue
            photo_value = torch.zeros_like(parameter) if photo is None else photo
            aux_value = torch.zeros_like(parameter) if aux is None else aux
            if self.last_projected:
                aux_value = aux_value - coefficient * photo_value
            parameter.grad = (photo_value + aux_value).detach()
