"""Discriminators for the conditional quantum / classical GANs.

Two variants live here:

* :class:`ConditionalDiscriminator` — the input-conditional discriminator
  shared by Pass-2 of the quantum notebook and the classical notebook. The
  class label is fed *into* D via an embedding concatenated with the
  flattened image.
* :class:`ConditionalDiscriminatorAC` — the AC-GAN style discriminator
  introduced in Pass-3 of the quantum notebook. The class label is *not*
  consumed as an input; instead D has a second linear head that predicts
  the class as supervision (cross-entropy). This forces D to learn
  class-discriminative features and provides a sharper class-conditional
  gradient for G than label-as-input alone.

Reference for AC-GAN:
    Odena, Olah & Shlens (2017), "Conditional Image Synthesis with
    Auxiliary Classifier GANs", https://arxiv.org/abs/1610.09585
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from dataset import IMAGE_SIZE, N_CLASSES


class ConditionalDiscriminator(nn.Module):
    """Class-conditional fully connected discriminator.

    Architecture::

        x   -> flatten     ┐
                           │ concat -> Linear(64+e, 64) -> ReLU
        y   -> Embedding(e)┘                            -> Linear(64, 16) -> ReLU
                                                        -> Linear(16, 1)  -> Sigmoid
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        image_size: int = IMAGE_SIZE,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size + embed_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.dtype != torch.long:
            y = y.long()
        h = torch.cat([x, self.label_emb(y)], dim=1)
        return self.model(h)


class ConditionalDiscriminatorAC(nn.Module):
    """AC-GAN style discriminator with a real/fake head and a class head.

    Architecture::

        x -> flatten -> Linear(64, 64) -> LeakyReLU -> Linear(64, 16) -> LeakyReLU
                                                       │
                                                       ├── Linear(16, 1) -> Sigmoid   (real/fake)
                                                       └── Linear(16, n_classes)      (class logits)

    The label is *not* fed as input. Instead the class head is trained
    against the true class via cross-entropy on both real and fake
    samples; G is trained against the same class head so that its
    fakes match the requested class. This replaces the
    ``match-aware`` mismatch loss used in Pass-2.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        image_size: int = IMAGE_SIZE,
        hidden_dim: int = 64,
        feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        self.shared = nn.Sequential(
            nn.Linear(image_size * image_size, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.real_fake_head = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        self.class_head = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.shared(x)
        return self.real_fake_head(h).view(-1), self.class_head(h)


# Backwards-compatible alias so `from discriminator import Discriminator` still
# works in any cells that may have been pre-existing notebooks. The default
# behaviour is now conditional.
Discriminator = ConditionalDiscriminator
