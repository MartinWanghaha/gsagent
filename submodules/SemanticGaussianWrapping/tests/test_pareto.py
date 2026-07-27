import torch

from regularization.pareto import PhotometricParetoGuard


def test_conflicting_auxiliary_gradient_is_projected() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    head = torch.nn.Parameter(torch.tensor(3.0))
    photo = parameter.sum()
    auxiliary = -parameter.sum() + head.square()
    guard = PhotometricParetoGuard(enabled=True)
    guard.backward(photo, auxiliary, [parameter])
    assert guard.last_projected
    # Exact opposite auxiliary direction is removed, retaining photo gradient.
    assert torch.allclose(parameter.grad, torch.ones_like(parameter))
    assert torch.allclose(head.grad, torch.tensor(6.0))


def test_non_conflicting_gradients_are_added() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    photo = parameter.sum()
    auxiliary = parameter.square().sum()
    PhotometricParetoGuard(enabled=True).backward(photo, auxiliary, [parameter])
    assert torch.allclose(parameter.grad, torch.tensor([3.0, 5.0]))
