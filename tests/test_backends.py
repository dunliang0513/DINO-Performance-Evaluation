import numpy as np
import pytest
import torch

from backends.base import l2_normalise, pool_tokens


def test_mean_pooling_excludes_prefix_tokens():
    """The spec's key correctness rule.

    DINOv2 has 1 prefix token (CLS); DINOv3 has CLS plus register tokens.
    Averaging any of them into the patch mean corrupts every embedding, and
    would do so silently, so it is pinned by a test here.
    """
    tokens = torch.zeros(1, 7, 4)
    tokens[:, :3] = 999.0   # 3 prefix tokens: CLS + 2 registers
    tokens[:, 3:] = 2.0     # 4 real patch tokens

    pooled = pool_tokens(tokens, pooling="mean", num_prefix_tokens=3)

    assert torch.allclose(pooled, torch.full((1, 4), 2.0)), (
        "prefix tokens leaked into the mean"
    )


def test_cls_pooling_takes_the_first_token():
    tokens = torch.zeros(1, 5, 3)
    tokens[:, 0] = 7.0
    tokens[:, 1:] = 1.0

    pooled = pool_tokens(tokens, pooling="cls", num_prefix_tokens=1)

    assert torch.allclose(pooled, torch.full((1, 3), 7.0))


def test_unknown_pooling_raises():
    tokens = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match="pooling"):
        pool_tokens(tokens, pooling="max", num_prefix_tokens=1)


def test_l2_normalise_produces_unit_vectors():
    features = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    normalised = l2_normalise(features)

    norms = torch.linalg.norm(normalised, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_l2_normalise_survives_a_zero_vector():
    features = torch.zeros(1, 4)
    normalised = l2_normalise(features)
    assert torch.isfinite(normalised).all(), "a zero vector must not produce NaN"
