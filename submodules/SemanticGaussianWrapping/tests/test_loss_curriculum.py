from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from regularization.losses import SemanticLossSystem
from scene.gaussian_model import SemanticDecoder
from semantic.geometry_policy import GeometryEvidenceProjector


def _render_package():
    return {
        "render": torch.rand(3, 12, 12, requires_grad=True),
        "semantic": torch.rand(16, 12, 12, requires_grad=True),
    }


def _system(weights, **kwargs) -> SemanticLossSystem:
    return SemanticLossSystem(
        16,
        2,
        weights,
        SemanticDecoder(16, 2),
        **kwargs,
    )


def test_bootstrap_does_not_touch_missing_gaussian_auxiliary_api() -> None:
    system = _system(
        {
            "lambda_region_rgb": 0.0,
            "lambda_semantic": 1.0,
            "lambda_manifold": 1.0,
        },
    )
    camera = SimpleNamespace(original_image=torch.rand(3, 12, 12))
    bundle = system(
        _render_package(),
        camera,
        object(),
        {"region_rgb": 0.0, "semantic": 0.0, "manifold": 0.0},
    )
    assert bundle.auxiliary.item() == 0.0
    bundle.total.backward()


def test_all_ignore_semantics_has_zero_image_losses_for_custom_ignore_label() -> None:
    system = _system({"lambda_semantic": 1.0, "lambda_boundary": 1.0})
    camera = SimpleNamespace(
        original_image=torch.rand(3, 12, 12),
        semantic_ids=torch.full((12, 12), 255, dtype=torch.long),
        semantic_confidence=torch.zeros(12, 12),
        semantic_boundary=torch.zeros(12, 12),
        ignore_label=255,
    )
    result = system.semantic_image_losses(_render_package()["semantic"], camera)
    assert result["semantic"].item() == 0.0
    assert result["boundary"].item() == 0.0


def test_zero_confidence_pixels_do_not_affect_semantic_or_boundary_loss() -> None:
    torch.manual_seed(2)
    system = _system({"lambda_semantic": 1.0, "lambda_boundary": 1.0})
    confidence = torch.zeros(4, 4)
    confidence[0, 0] = 1.0
    camera = SimpleNamespace(
        semantic_ids=torch.zeros(4, 4, dtype=torch.long),
        semantic_confidence=confidence,
        semantic_boundary=torch.zeros(4, 4),
        ignore_label=-1,
    )
    first = torch.randn(16, 4, 4, requires_grad=True)
    second = first.detach().clone()
    second[:, 1:, :] += 100.0
    second[:, 0, 1:] -= 100.0
    second.requires_grad_()
    first_loss = sum(system.semantic_image_losses(first, camera).values())
    second_loss = sum(system.semantic_image_losses(second, camera).values())
    assert torch.allclose(first_loss, second_loss, atol=1e-6)
    first_loss.backward()
    assert torch.count_nonzero(first.grad[:, 1:, :]) == 0
    assert torch.count_nonzero(first.grad[:, 0, 1:]) == 0


def test_region_rgb_is_pareto_guarded_not_folded_into_global_photo() -> None:
    system = _system(
        {"lambda_region_rgb": 2.0, "region_area_temperature": 0.5},
    )
    package = _render_package()
    camera = SimpleNamespace(
        original_image=torch.zeros(3, 12, 12),
        semantic_ids=torch.ones(12, 12, dtype=torch.long),
        semantic_confidence=torch.ones(12, 12),
    )

    global_photo, _ = system.photometric_loss(
        package["render"],
        camera.original_image,
    )
    region_photo = system.region_photometric_loss(
        package["render"],
        camera.original_image,
        camera,
    )
    bundle = system(package, camera, object(), {"region_rgb": 1.0})

    assert torch.allclose(bundle.photometric, global_photo)
    assert torch.allclose(bundle.auxiliary, 2.0 * region_photo)


def test_geometry_evidence_checkpoint_restores_propagation_cursor() -> None:
    projector = GeometryEvidenceProjector()
    projector._propagation_cursor.fill_(12_345)
    system = _system(
        {},
        evidence_projector=projector,
    )
    state = system.evidence_state_dict()

    restored_projector = GeometryEvidenceProjector()
    restored = _system(
        {},
        evidence_projector=restored_projector,
    )
    restored.load_evidence_state_dict(state)
    assert int(restored_projector._propagation_cursor) == 12_345


def test_geometry_evidence_checkpoint_rejects_implicit_empty_state() -> None:
    system = _system({}, evidence_projector=GeometryEvidenceProjector())
    with pytest.raises(ValueError, match="schema must be version 1"):
        system.load_evidence_state_dict({})


def test_chunked_semantic_image_loss_is_exact() -> None:
    decoder = SemanticDecoder(16, 5, temperature=0.2)
    system = SemanticLossSystem(
        16,
        5,
        {},
        decoder,
        region_decode_chunk_size=7,
    )
    embedding = torch.randn(16, 4, 5, requires_grad=True)
    ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, -1, 0, 1],
            [3, 4, 0, 1, 2],
        ]
    )
    confidence = torch.linspace(0.1, 1.0, 20).reshape(4, 5)
    camera = SimpleNamespace(
        semantic_ids=ids,
        semantic_confidence=confidence,
        semantic_boundary=None,
        ignore_label=-1,
    )

    actual = system.semantic_image_losses(embedding, camera)["semantic"]
    actual_gradients = torch.autograd.grad(
        actual,
        (embedding, decoder.linear.weight, decoder.linear.bias),
    )

    reference_embedding = embedding.detach().clone().requires_grad_(True)
    logits = system.classifier(reference_embedding[None])[0]
    per_pixel = F.cross_entropy(
        logits[None],
        ids[None],
        ignore_index=-1,
        reduction="none",
    )[0]
    valid_weight = confidence * (ids != -1)
    expected = (per_pixel * valid_weight).sum() / valid_weight.sum()
    expected_gradients = torch.autograd.grad(
        expected,
        (reference_embedding, decoder.linear.weight, decoder.linear.bias),
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        assert torch.allclose(
            actual_gradient,
            expected_gradient,
            atol=1e-6,
            rtol=1e-6,
        )
    assert torch.allclose(
        system.semantic_residual(embedding.detach(), camera),
        per_pixel.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
