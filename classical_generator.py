"""Classical conditional generators used as a baseline against the patch QGAN.

Two flavours are provided:

* :class:`ConditionalClassicalGenerator` -- a standard conditional MLP
  generator. Concatenates a label embedding with the latent vector and feeds
  the result through a two-hidden-layer MLP.
* :class:`ConditionalTinyGenerator` -- a deliberately bottlenecked variant
  whose total parameter count is comparable to the patch quantum generator
  for a like-for-like capacity comparison.

Both generators output sigmoid-bounded ``[0, 1]`` pixel intensities so that
they live in the same space as the quantum generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dataset import IMAGE_SIZE, N_CLASSES


class ConditionalClassicalGenerator(nn.Module):
    """Standard conditional MLP generator (latent + label -> image).

    Parameters
    ----------
    latent_dim:
        Size of the latent vector ``z``.
    n_classes:
        Number of conditioning classes.
    hidden_dim:
        Width of the hidden layers.
    image_size:
        Side length of the (square) output image.
    embed_dim:
        Dimensionality of the label embedding.
    """

    def __init__(
        self,
        latent_dim: int = 5,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 64,
        image_size: int = IMAGE_SIZE,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, image_size * image_size),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype != torch.long:
            y = y.long()
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.model(h)


class ConditionalTinyGenerator(nn.Module):
    """Conditional MLP generator with ~quantum-comparable parameter budget.

    Default configuration produces ``≈ 234`` trainable parameters
    (``50`` for the embedding + ``≈ 184`` for the MLP) which is close to the
    ``170`` parameters of the conditional patch quantum generator.
    """

    def __init__(
        self,
        latent_dim: int = 5,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 1,
        image_size: int = IMAGE_SIZE,
        embed_dim: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, image_size * image_size),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype != torch.long:
            y = y.long()
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.model(h)


# Backwards-compatible aliases.
ClassicalGenerator = ConditionalClassicalGenerator
TinyClassicalGenerator = ConditionalTinyGenerator
