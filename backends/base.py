"""The feature-extractor contract shared by every backend."""

from typing import Protocol

import torch


def pool_tokens(tokens, pooling, num_prefix_tokens):
    """Reduce a token sequence [B, T, D] to one vector per item, [B, D].

    `num_prefix_tokens` counts the non-patch tokens at the front of the
    sequence: 1 for DINOv2 (CLS), and CLS plus the register tokens for DINOv3.
    Mean pooling must skip all of them.
    """
    if pooling == "cls":
        return tokens[:, 0]

    if pooling == "mean":
        return tokens[:, num_prefix_tokens:].mean(dim=1)

    raise ValueError(
        f"Unknown pooling {pooling!r}; expected 'cls' or 'mean'."
    )


def l2_normalise(features, epsilon=1e-12):
    """Scale each row to unit length so cosine similarity is a dot product.

    Clamping the norm keeps an all-zero row (a fully black crop) from producing
    NaNs that would poison every downstream score.
    """
    norms = torch.linalg.norm(features, dim=1, keepdim=True).clamp_min(epsilon)
    return features / norms


class FeatureExtractor(Protocol):
    """What every backend must provide."""

    name: str
    dim: int
    num_prefix_tokens: int

    def embed(self, crops_bgr, pooling="cls"):
        """Embed a list of BGR uint8 crops, returning [N, dim] unit-norm float32."""
        ...

    def warmup(self, batch_size):
        """Run throwaway inferences so later timings exclude lazy setup."""
        ...
