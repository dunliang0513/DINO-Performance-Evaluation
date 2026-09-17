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


# --- Backend integration tests -------------------------------------------
# These download weights on first run (roughly 90 MB for small, 350 MB for
# base), so they are marked and excluded from the default run.

MODEL_DIMS = {
    "dinov2-small": 384,
    "dinov2-base": 768,
    "dinov3-small": 384,
    "dinov3-base": 768,
}


@pytest.mark.download
@pytest.mark.parametrize("model_name,expected_dim", sorted(MODEL_DIMS.items()))
def test_embeddings_have_the_right_shape_and_norm(
    model_name, expected_dim, synthetic_board
):
    from backends import get_backend

    backend = get_backend(model_name, device="cpu", use_fp16=False)
    crops = [
        synthetic_board[100:180, 100:180],
        synthetic_board[300:380, 300:380],
    ]

    features = backend.embed(crops, pooling="cls")

    assert features.shape == (2, expected_dim)
    assert backend.dim == expected_dim
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-4)
    assert features.dtype == np.float32


@pytest.mark.download
def test_dinov3_reports_register_tokens(synthetic_board):
    """DINOv3 carries register tokens beyond the CLS token.

    If this ever reports 1, mean pooling silently averages register tokens into
    the patch mean, so it is asserted rather than assumed.
    """
    from backends import get_backend

    backend = get_backend("dinov3-small", device="cpu", use_fp16=False)
    assert backend.num_prefix_tokens > 1


@pytest.mark.download
def test_pooling_modes_give_different_embeddings(synthetic_board):
    from backends import get_backend

    backend = get_backend("dinov2-small", device="cpu", use_fp16=False)
    crops = [synthetic_board[100:180, 100:180]]

    cls_features = backend.embed(crops, pooling="cls")
    mean_features = backend.embed(crops, pooling="mean")

    assert not np.allclose(cls_features, mean_features), (
        "cls and mean pooling should not collapse to the same vector"
    )


@pytest.mark.download
def test_identical_crops_are_maximally_similar(synthetic_board):
    """Sanity check on the similarity metric the whole pipeline rests on."""
    from backends import get_backend

    backend = get_backend("dinov2-small", device="cpu", use_fp16=False)
    crop = synthetic_board[100:180, 100:180]

    features = backend.embed([crop, crop.copy()], pooling="cls")
    similarity = float(features[0] @ features[1])

    assert similarity > 0.999
