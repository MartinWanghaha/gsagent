from types import SimpleNamespace

import torch

from mesh import ContactGraph


def test_contact_graph_is_learned_from_confident_covariance_overlap() -> None:
    count = 12
    xyz = torch.zeros(count, 3)
    xyz[:, 0] = torch.arange(count) * 0.01
    labels = torch.arange(count) % 2
    semantic = torch.nn.functional.one_hot(labels, 2).float()
    gaussians = SimpleNamespace(
        get_xyz=xyz,
        get_scaling=torch.full((count, 3), 0.08),
        get_semantic_confidence=torch.ones(count, 1),
        get_semantic=semantic,
        semantic_decoder=lambda value: value,
    )
    graph = ContactGraph.from_gaussians(
        gaussians,
        threshold=0.1,
        neighbors=4,
        min_support=1.0,
        background_id=None,
    )
    assert graph.allows(0, 1)
    assert graph.scores[(0, 1)] > 0.1


def test_contact_graph_requires_a_semantic_decoder() -> None:
    gaussians = SimpleNamespace(
        get_xyz=torch.rand(4, 3),
        get_scaling=torch.ones(4, 3),
        get_semantic_confidence=torch.ones(4, 1),
        get_semantic=torch.rand(4, 16),
        semantic_decoder=None,
    )
    assert ContactGraph.from_gaussians(gaussians).scores == {}
