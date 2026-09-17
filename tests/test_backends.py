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


# --- Production-path tests -------------------------------------------------
# Everything above runs on CPU in float32 for determinism, but the shipped
# defaults are DEVICE=cuda and USE_FP16=True. Two silent-corruption bugs
# survive the CPU-only tests, so they are pinned here against the real
# configuration.

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.mark.download
@requires_cuda
def test_fp16_forward_still_yields_float32_unit_norm():
    """Guards the `.float()` cast before normalisation.

    The forward pass runs in fp16, but normalising in fp16 loses precision
    near unit length. With the cast removed the norms drift far enough to
    matter against a 0.75 similarity threshold, yet no CPU test notices
    because use_fp16=False makes the cast a no-op there.
    """
    from backends import get_backend

    backend = get_backend("dinov2-small", device="cuda", use_fp16=True)
    rng = np.random.default_rng(11)
    crops = [rng.integers(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(4)]

    features = backend.embed(crops, pooling="cls")

    assert features.dtype == np.float32
    norms = np.linalg.norm(features, axis=1)
    # fp32 normalisation lands within ~1e-6; fp16 normalisation is ~100x worse.
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms drifted: {norms}"


@pytest.mark.download
@pytest.mark.parametrize("model_name", ["dinov2-small", "dinov3-small"])
def test_crops_are_converted_from_bgr_to_rgb(model_name):
    """Guards the cv2.COLOR_BGR2RGB conversion.

    OpenCV hands us BGR; both backbones expect RGB. Dropping the conversion
    degrades every embedding identically on both sides of the comparison, so
    nothing downstream crashes or looks wrong -- it would just quietly
    invalidate the DINOv2-vs-DINOv3 accuracy numbers this repo exists to
    produce. Asserted on the preprocessed tensor rather than the embedding,
    because that is where the ordering is observable.
    """
    from backends import get_backend

    backend = get_backend(model_name, device="cpu", use_fp16=False)

    # Strongly blue-dominant in BGR: channel 0 is B=230, channel 2 is R=10.
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    crop[:, :, 0] = 230
    crop[:, :, 1] = 40
    crop[:, :, 2] = 10

    pixel_values = backend._to_pixel_values([crop])

    red_channel = float(pixel_values[0, 0].mean())
    blue_channel = float(pixel_values[0, 2].mean())

    assert red_channel < blue_channel, (
        f"{model_name}: expected the RGB red channel ({red_channel:.3f}) to be "
        f"darker than blue ({blue_channel:.3f}) for a blue-dominant BGR crop; "
        f"BGR->RGB conversion is probably missing"
    )
