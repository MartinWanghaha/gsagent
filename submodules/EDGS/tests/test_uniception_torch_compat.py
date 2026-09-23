from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from source.vendor import bootstrap_mvroma  # noqa: E402

bootstrap_mvroma()

from uniception.models.utils.transformer_blocks import (  # noqa: E402
    Attention,
    _scaled_dot_product_attention,
)
from uniflowmatch.models.unet_encoder import _resize_nearest_2d  # noqa: E402


def test_explicit_sdpa_scale_matches_reference_on_torch_20():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    key = torch.randn(2, 3, 6, 4, dtype=torch.float64)
    value = torch.randn(2, 3, 6, 8, dtype=torch.float64)
    scale = 0.25

    actual = _scaled_dot_product_attention(query, key, value, scale=scale)
    weights = torch.softmax(
        query @ key.transpose(-2, -1) * scale,
        dim=-1,
    )
    expected = weights @ value

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_uniception_fused_attention_runs_in_paintmesh_torch():
    attention = Attention(dim=32, num_heads=4).eval()
    inputs = torch.randn(2, 11, 32)

    with torch.inference_mode():
        output = attention(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_ufm_nearest_resize_preserves_bfloat16_values():
    inputs = torch.randn(1, 3, 5, 7).to(torch.bfloat16)

    actual = _resize_nearest_2d(inputs, size=(9, 11))
    expected = torch.nn.functional.interpolate(
        inputs.float(), size=(9, 11), mode="nearest"
    ).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
