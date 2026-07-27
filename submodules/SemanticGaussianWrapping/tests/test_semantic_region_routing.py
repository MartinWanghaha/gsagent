from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from regularization.losses import PixelSemanticDecoder
from scene.dataset_readers import BasicPointCloud
from scene.gaussian_model import GaussianModel, SemanticDecoder
from semantic.region_membership import (
    SparseRegionMembership,
    decode_sparse_region_memberships,
)


class RecordingDecoder(nn.Module):
    def __init__(self, semantic_dim: int, num_classes: int) -> None:
        super().__init__()
        self.decoder = SemanticDecoder(semantic_dim, num_classes, temperature=0.13)
        self.calls: list[tuple[torch.Size, torch.dtype, torch.dtype]] = []

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(embedding)
        self.calls.append((embedding.shape, embedding.dtype, logits.dtype))
        return logits


def _decode(
    decoder,
    embedding,
    indices,
    *,
    num_classes,
    top_k,
    chunk_size,
    confidence=None,
):
    return decode_sparse_region_memberships(
        embedding,
        indices,
        decoder=decoder,
        num_classes=num_classes,
        top_k=top_k,
        chunk_size=chunk_size,
        confidence=confidence,
    )


def test_from_logits_keeps_exact_foreground_topk_without_renormalizing() -> None:
    logits = torch.tensor(
        [[2.0, -1.0, 3.0, 0.5], [-2.0, 0.0, 1.0, 4.0]],
        dtype=torch.float16,
    )
    confidence = torch.tensor([[0.25], [0.75]], dtype=torch.float16)
    actual = SparseRegionMembership.from_logits(logits, top_k=2, confidence=confidence)
    probability = torch.softmax(logits.float(), dim=1)
    expected_weights, expected_local_ids = probability[:, 1:].topk(2, dim=1)

    assert torch.equal(actual.ids, expected_local_ids + 1)
    assert torch.equal(actual.weights, expected_weights)
    assert torch.equal(actual.background, probability[:, :1])
    assert torch.allclose(
        actual.tail,
        1.0 - actual.background - actual.weights.sum(dim=1, keepdim=True),
    )
    assert torch.equal(actual.confidence, confidence.float())
    assert torch.allclose(actual.foreground_mass, 1.0 - actual.background)
    assert torch.allclose(actual.probability(0), actual.background)
    assert torch.allclose(
        actual.probability(torch.tensor([2, 3], dtype=torch.long)),
        torch.stack((probability[:, 2], probability[:, 3]), dim=1),
    )


def test_membership_index_select_preserves_every_field() -> None:
    membership = SparseRegionMembership.from_logits(torch.randn(5, 7), top_k=3)
    indices = torch.tensor([4, 1, 1], dtype=torch.long)
    selected = membership.index_select(indices)
    for name in ("ids", "weights", "background", "tail", "confidence"):
        assert torch.equal(getattr(selected, name), getattr(membership, name)[indices])

    converted = selected.to(dtype=torch.float16)
    assert converted.ids.dtype == torch.long
    for name in ("weights", "background", "tail", "confidence"):
        assert getattr(converted, name).dtype == torch.float16


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ({"ids": torch.zeros(2, 2, dtype=torch.long)}, ValueError),
        ({"weights": torch.ones(2, 2, dtype=torch.float64)}, TypeError),
        ({"background": torch.zeros(2)}, ValueError),
        ({"tail": torch.full((2, 1), 0.8)}, ValueError),
        ({"confidence": torch.full((2, 1), 1.1)}, ValueError),
    ],
)
def test_membership_strictly_validates_contract(replacement, error) -> None:
    valid = SparseRegionMembership.from_logits(torch.randn(2, 4), top_k=2)
    values = {
        name: getattr(valid, name)
        for name in ("ids", "weights", "background", "tail", "confidence")
    }
    values.update(replacement)
    with pytest.raises(error):
        SparseRegionMembership(**values)


def test_chunked_candidate_memberships_equal_dense_fp32_topk() -> None:
    torch.manual_seed(1729)
    semantic_dim, num_classes = 16, 23
    decoder = RecordingDecoder(semantic_dim, num_classes)
    classifier = PixelSemanticDecoder(semantic_dim, num_classes, decoder)
    embedding = torch.randn(257, semantic_dim)
    indices = torch.tensor([251, 3, 89, 17, 129, 6, 203, 41, 77, 1])
    confidence = torch.linspace(0.0, 1.0, len(embedding))[:, None]

    routed_embedding = embedding.half().float()
    dense_logits = classifier(routed_embedding.T[None, :, :, None])[0, :, :, 0].T
    expected = SparseRegionMembership.from_logits(
        dense_logits.index_select(0, indices),
        top_k=4,
        confidence=confidence.index_select(0, indices),
    )
    decoder.calls.clear()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = _decode(
            decoder,
            embedding.half(),
            indices,
            num_classes=num_classes,
            top_k=4,
            chunk_size=3,
            confidence=confidence,
        )

    assert torch.equal(actual.ids, expected.ids)
    assert torch.equal(actual.confidence, expected.confidence)
    for name in ("weights", "background", "tail"):
        assert torch.allclose(
            getattr(actual, name), getattr(expected, name), atol=2e-7, rtol=2e-6
        )
    assert [shape[0] for shape, _, _ in decoder.calls] == [3, 3, 3, 1]
    assert all(input_dtype == torch.float32 for _, input_dtype, _ in decoder.calls)
    assert all(logit_dtype == torch.float32 for _, _, logit_dtype in decoder.calls)


def test_empty_decode_and_background_only_class_have_explicit_shapes() -> None:
    decoder = RecordingDecoder(16, 1)
    result = _decode(
        decoder,
        torch.randn(20, 16),
        torch.empty(0, dtype=torch.long),
        num_classes=1,
        top_k=4,
        chunk_size=4,
    )

    assert result.ids.shape == result.weights.shape == (0, 0)
    assert result.background.shape == result.tail.shape == result.confidence.shape == (0, 1)
    assert decoder.calls == []

    background_only = SparseRegionMembership.from_logits(torch.randn(3, 1), top_k=7)
    assert background_only.ids.shape == background_only.weights.shape == (3, 0)
    assert torch.equal(background_only.background, torch.ones(3, 1))
    assert torch.equal(background_only.tail, torch.zeros(3, 1))


@pytest.mark.parametrize("chunk_size", [0, -1, True])
def test_candidate_decode_rejects_invalid_chunk_size(chunk_size) -> None:
    decoder = SemanticDecoder(16, 7)
    with pytest.raises(ValueError, match="chunk_size"):
        _decode(
            decoder,
            torch.randn(20, 16),
            torch.tensor([1, 2], dtype=torch.long),
            num_classes=7,
            top_k=2,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_candidate_decode_rejects_invalid_top_k(top_k) -> None:
    decoder = SemanticDecoder(16, 7)
    with pytest.raises(ValueError, match="top_k"):
        _decode(
            decoder,
            torch.randn(20, 16),
            torch.tensor([1, 2], dtype=torch.long),
            num_classes=7,
            top_k=top_k,
            chunk_size=4,
        )


def test_candidate_decode_validates_index_and_confidence_contract() -> None:
    decoder = SemanticDecoder(16, 7)
    embedding = torch.randn(20, 16)
    with pytest.raises(TypeError, match="torch.long"):
        _decode(
            decoder,
            embedding,
            torch.tensor([1, 2], dtype=torch.int32),
            num_classes=7,
            top_k=2,
            chunk_size=4,
        )
    with pytest.raises(ValueError, match=r"shape \[M\]"):
        _decode(
            decoder,
            embedding,
            torch.tensor([[1, 2]], dtype=torch.long),
            num_classes=7,
            top_k=2,
            chunk_size=4,
        )
    with pytest.raises(IndexError, match="out of range"):
        _decode(
            decoder,
            embedding,
            torch.tensor([1, 20], dtype=torch.long),
            num_classes=7,
            top_k=2,
            chunk_size=4,
        )
    with pytest.raises(ValueError, match=r"shape \[N,1\]"):
        _decode(
            decoder,
            embedding,
            torch.tensor([1, 2], dtype=torch.long),
            num_classes=7,
            top_k=2,
            chunk_size=4,
            confidence=torch.ones(20),
        )


def test_gaussian_model_memberships_use_combined_evidence_and_round_trip() -> None:
    cloud = BasicPointCloud(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        colors=np.full((3, 3), 0.5, dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
    )
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.configure_semantic_decoder(5, temperature=0.2)
    with torch.no_grad():
        model.registry["semantic_confidence"].copy_(torch.tensor([[0.2], [0.8], [0.1]]))
        model.registry["propagated_semantic_confidence"].copy_(
            torch.tensor([[0.7], [0.3], [0.4]])
        )
    indices = torch.tensor([2, 0], dtype=torch.long)
    expected = model.point_region_memberships(indices, top_k=2, chunk_size=1)
    assert torch.equal(expected.confidence, torch.tensor([[0.4], [0.7]]))

    restored = GaussianModel(1, semantic_dim=4, device="cpu")
    restored.restore(model.capture())
    actual = restored.point_region_memberships(indices, top_k=2, chunk_size=8)
    assert torch.equal(actual.ids, expected.ids)
    assert torch.equal(actual.confidence, expected.confidence)
    for name in ("weights", "background", "tail"):
        assert torch.allclose(
            getattr(actual, name), getattr(expected, name), atol=2e-7, rtol=2e-6
        )


def test_gaussian_model_membership_requires_decoder() -> None:
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    with pytest.raises(RuntimeError, match="configure_semantic_decoder"):
        model.point_region_memberships(
            torch.empty(0, dtype=torch.long), top_k=2, chunk_size=4
        )
