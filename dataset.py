"""Dataset utilities for the handwritten-digits Conditional GAN.

The Optical Recognition of Handwritten Digits dataset (UCI) ships with 8x8
grayscale images flattened into rows of a CSV-style file (``optdigits.tra``).
Each row contains 64 pixel intensities in the range ``[0, 16]`` followed by an
integer class label in the final column.

:class:`DigitsDataset` returns ``(image, label)`` pairs for *all* 10 digit
classes by default, which is what the conditional GAN consumes during
training. Pass ``label=k`` to filter down to a single class (e.g. for
evaluation against a specific digit).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGE_SIZE = 8
N_CLASSES = 10


def load_full_digits(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return all UCI handwritten-digit samples and their labels.

    Used by the evaluation pipeline to train the multi-class "judge" and the
    domain-specific feature extractor.

    Returns
    -------
    X:
        Float32 array of shape ``(N, 1, 8, 8)`` with pixels in ``[0, 1]``.
    y:
        Int64 array of shape ``(N,)`` with class labels in ``{0, ..., 9}``.
    """
    df = pd.read_csv(csv_file, header=None)
    pixels = df.iloc[:, :-1].to_numpy(dtype=np.float32) / 16.0
    labels = df.iloc[:, -1].to_numpy(dtype=np.int64)
    images = pixels.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    return images, labels


class DigitsDataset(Dataset):
    """PyTorch dataset for the UCI Optical Recognition of Handwritten Digits.

    Parameters
    ----------
    csv_file:
        Path to ``optdigits.tra``. Each row is 64 pixel values in ``[0, 16]``
        followed by an integer label.
    label:
        If ``None`` (default) keep all 10 classes; otherwise filter to that
        class only.
    transform:
        Optional callable applied to each ``(8, 8)`` ``np.ndarray`` sample
        (e.g. ``torchvision.transforms.ToTensor``).
    """

    def __init__(
        self,
        csv_file: str,
        label: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_file = csv_file
        self.transform = transform
        self.df = self._filter_by_label(label)

    def _filter_by_label(self, label: Optional[int]) -> pd.DataFrame:
        df = pd.read_csv(self.csv_file, header=None)
        if label is not None:
            df = df.loc[df.iloc[:, -1] == label]
        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        image = row.iloc[:-1].to_numpy(dtype=np.float32) / 16.0
        image = image.reshape(IMAGE_SIZE, IMAGE_SIZE)
        label = int(row.iloc[-1])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
