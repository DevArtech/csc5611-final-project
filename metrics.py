"""Quantitative evaluation utilities for the conditional GANs.

The same metrics are reused across :mod:`quantum_gan.ipynb` and
:mod:`classical_gan.ipynb` so the two models are compared on identical
yardsticks.

Generators in this project all expose a ``forward(z, y)`` interface where
``z`` is a latent tensor and ``y`` is an integer class-label tensor. The
metrics here adopt a generic ``sample_inputs(n) -> tuple`` callable so that
both unconditional (``(z,)``) and conditional (``(z, y)``) generators can be
plugged in without changes -- the helper :func:`call_generator` simply
unpacks the tuple.

Provided utilities
------------------

``preprocess_for_inception``
    Resize 8x8 grayscale images to ``299x299`` RGB ``uint8`` tensors so they
    can be fed to ``torchmetrics.image.inception.InceptionScore``.

``calculate_frechet_distance``
    Numerically stable Fréchet distance between two diagonal-augmented
    Gaussians.

``DigitFeatureExtractor`` / ``train_feature_extractor``
    64-dim feature embedding network trained as a 10-way digit classifier.

``train_judge``
    Random-Forest classifier trained on the *full* multi-class dataset.

``compute_inception_score``
    Standard ImageNet Inception Score via ``torchmetrics``. Reported for
    completeness even though it is essentially uninformative on 8x8 digits.

``compute_custom_fid``
    Domain-specific FID between a batch of real samples and a batch of
    generator samples in the ``DigitFeatureExtractor`` feature space. Set
    ``per_class=True`` (or use :func:`compute_per_class_fid`) for the
    breakdown across all 10 classes.

``compute_judge_metrics``
    Class-confusion matrix, marginal class distribution, judge inception
    score, per-class accuracy, and overall accuracy (from a 10-class judge).

``compute_pixel_diversity``
    Mean pairwise L2 distance + per-pixel std as a coarse intra-class
    diversity proxy.

``measure_generation_speed``
    Wall-clock samples-per-second throughput.

``interpolate_points``
    Linear interpolation between two latent vectors.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import sqrtm
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from dataset import IMAGE_SIZE, N_CLASSES


SampleInputs = Callable[[int], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]


def _ensure_tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x,)


def call_generator(
    generator: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Unwrap ``inputs`` (single tensor or tuple) and call ``generator``."""
    return generator(*_ensure_tuple(inputs))


# ---------------------------------------------------------------------------
# Preprocessing for the standard Inception Score
# ---------------------------------------------------------------------------

def preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Resize 8x8 grayscale images to ``(N, 3, 299, 299)`` uint8 RGB tensors."""
    if images.dim() == 2:
        images = images.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    images = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )
    images = images.repeat(1, 3, 1, 1)
    if images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0.0, 1.0)
    return (images * 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Domain-specific FID utilities
# ---------------------------------------------------------------------------

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Numerically-stable Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    ssdiff = float(np.sum(diff ** 2))

    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(
                f"calculate_frechet_distance: large imaginary component "
                f"{np.max(np.abs(covmean.imag)):.3e}, taking real part."
            )
        covmean = covmean.real

    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


class DigitFeatureExtractor(nn.Module):
    """Tiny MLP feature extractor with an attached classifier head."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.backbone(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward(x))


def train_feature_extractor(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> DigitFeatureExtractor:
    """Train ``DigitFeatureExtractor`` as a 10-way digit classifier."""
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
    )

    extractor = DigitFeatureExtractor(n_classes=int(y.max()) + 1).to(device)
    optimizer = optim.Adam(extractor.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    extractor.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = extractor.classify(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
        if verbose:
            print(
                f"  feature extractor epoch {epoch + 1:2d}/{epochs} | "
                f"loss {total_loss / total:.4f} | acc {correct / total:.3f}"
            )
    extractor.eval()
    return extractor


def train_judge(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Random-Forest classifier trained on the full multi-class dataset."""
    X_flat = X.reshape(X.shape[0], -1)
    judge = RandomForestClassifier(n_jobs=-1, random_state=0)
    judge.fit(X_flat, y)
    return judge


# ---------------------------------------------------------------------------
# Generator-based metrics
# ---------------------------------------------------------------------------

def _generate_batches(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    return_labels: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate ``n_samples`` images produced by ``generator``.

    If ``return_labels=True`` (only meaningful for conditional generators),
    also collects the class labels emitted by ``sample_inputs``. The labels
    are assumed to be the *second* element returned by ``sample_inputs``.
    """
    generator.eval()
    out = []
    label_out = []
    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            n = min(batch_size, remaining)
            inputs = sample_inputs(n)
            inputs_t = _ensure_tuple(inputs)
            inputs_t = tuple(t.to(device) for t in inputs_t)
            imgs = generator(*inputs_t)
            if imgs.dim() == 2:
                imgs = imgs.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            out.append(imgs.cpu())
            if return_labels:
                if len(inputs_t) < 2:
                    raise ValueError(
                        "return_labels=True requires sample_inputs to return "
                        "at least (z, y)."
                    )
                label_out.append(inputs_t[1].cpu())
            remaining -= n
    images = torch.cat(out, dim=0)[:n_samples]
    if return_labels:
        labels = torch.cat(label_out, dim=0)[:n_samples]
        return images, labels
    return images


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

def compute_inception_score(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    device: torch.device,
    n_samples: int = 1024,
    batch_size: int = 32,
    feature: int = 64,
    splits: int = 10,
) -> Tuple[float, float]:
    """Standard ImageNet-Inception Score via ``torchmetrics``."""
    from torchmetrics.image.inception import InceptionScore

    metric = InceptionScore(feature=feature, splits=splits, normalize=False).to(device)
    fakes = _generate_batches(generator, sample_inputs, n_samples, batch_size, device)
    for start in range(0, fakes.size(0), batch_size):
        chunk = fakes[start : start + batch_size].to(device)
        metric.update(preprocess_for_inception(chunk))
    is_mean, is_std = metric.compute()
    return float(is_mean.item()), float(is_std.item())


# ---------------------------------------------------------------------------
# Domain-specific FID (overall + per-class)
# ---------------------------------------------------------------------------

def _features(
    extractor: DigitFeatureExtractor,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    feats = []
    extractor.eval()
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            chunk = images[start : start + batch_size].to(device)
            feats.append(extractor(chunk).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_custom_fid(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    real_images: torch.Tensor,
    extractor: DigitFeatureExtractor,
    device: torch.device,
    n_samples: int = 1024,
    batch_size: int = 64,
) -> float:
    """Aggregate domain-specific FID between ``real_images`` and fakes."""
    fakes = _generate_batches(generator, sample_inputs, n_samples, batch_size, device)

    real_feats = _features(extractor, real_images, device, batch_size)
    fake_feats = _features(extractor, fakes, device, batch_size)

    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    return calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)


def compute_per_class_fid(
    generator: nn.Module,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    extractor: DigitFeatureExtractor,
    device: torch.device,
    sample_latent: Callable[[int], torch.Tensor],
    n_samples_per_class: int = 200,
    batch_size: int = 64,
    n_classes: int = N_CLASSES,
) -> Dict[int, float]:
    """Per-class FID: for each class ``c`` compare real-c features against
    fake samples generated with ``y = c`` (using ``sample_latent`` for ``z``)."""
    fid_scores: Dict[int, float] = {}
    for c in range(n_classes):
        # Real features for class c
        mask = (real_labels == c)
        real_c = real_images[mask]
        if real_c.size(0) < 5:
            fid_scores[c] = float("nan")
            continue

        # Generate n_samples_per_class fakes conditioned on class c
        def class_inputs(n, _c=c):
            z = sample_latent(n).to(device)
            y = torch.full((n,), _c, dtype=torch.long, device=device)
            return (z, y)

        fakes_c = _generate_batches(
            generator, class_inputs, n_samples_per_class, batch_size, device
        )
        real_feats = _features(extractor, real_c, device, batch_size)
        fake_feats = _features(extractor, fakes_c, device, batch_size)
        if real_feats.shape[0] < 2 or fake_feats.shape[0] < 2:
            fid_scores[c] = float("nan")
            continue

        mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
        mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
        fid_scores[c] = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)
    return fid_scores


# ---------------------------------------------------------------------------
# Random-Forest "judge" metrics
# ---------------------------------------------------------------------------

def compute_judge_metrics(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    judge: RandomForestClassifier,
    device: torch.device,
    n_samples: int = 1000,
    batch_size: int = 64,
    n_classes: int = N_CLASSES,
) -> Dict[str, Any]:
    """Class confusion, accuracy, and IS computed from the 10-class judge.

    ``sample_inputs`` is expected to return ``(z, y)`` -- this lets the judge
    evaluate whether each conditional sample matches the *requested* class.

    Returns a dict with:

    * ``predicted_distribution`` -- length ``n_classes`` array of judge
      argmax counts.
    * ``confusion_matrix`` -- ``(n_classes, n_classes)`` int matrix indexed
      as ``confusion[requested, predicted]``.
    * ``per_class_accuracy`` -- length ``n_classes`` array of fractions of
      requests correctly produced.
    * ``overall_accuracy`` -- mean of ``per_class_accuracy``.
    * ``mean_target_confidence`` -- average ``p(y_pred = y_request | x)``.
    * ``judge_inception_score`` -- ``exp( E[KL(p(y|x) || p(y))] )``.
    """
    fakes, labels = _generate_batches(
        generator, sample_inputs, n_samples, batch_size, device,
        return_labels=True,
    )
    fakes_flat = fakes.view(fakes.size(0), -1).numpy()
    labels_np = labels.numpy()

    probs = judge.predict_proba(fakes_flat)
    classes = list(judge.classes_)
    preds = probs.argmax(axis=1)

    # Marginal class distribution and judge IS
    py = probs.mean(axis=0)
    kl_per_sample = np.array([entropy(probs[i], py) for i in range(probs.shape[0])])
    judge_is = float(np.exp(kl_per_sample.mean()))

    distribution = np.zeros(n_classes, dtype=np.int64)
    for c, count in zip(*np.unique(preds, return_counts=True)):
        if int(c) < n_classes:
            distribution[int(c)] = int(count)

    # Confusion matrix indexed as [requested, predicted]
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    pred_class_idx = np.array([classes[i] for i in preds])
    for req, pred in zip(labels_np, pred_class_idx):
        if 0 <= req < n_classes and 0 <= pred < n_classes:
            confusion[int(req), int(pred)] += 1

    per_class_acc = np.zeros(n_classes)
    for c in range(n_classes):
        total = confusion[c].sum()
        per_class_acc[c] = (confusion[c, c] / total) if total > 0 else float("nan")

    # Mean p(y = requested | x)
    target_conf = []
    for i, req in enumerate(labels_np):
        if req in classes:
            target_conf.append(probs[i, classes.index(int(req))])
    mean_target_conf = float(np.mean(target_conf)) if target_conf else 0.0

    overall = np.nanmean(per_class_acc)

    return {
        "predicted_distribution": distribution,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_acc,
        "overall_accuracy": float(overall),
        "mean_target_confidence": mean_target_conf,
        "judge_inception_score": judge_is,
    }


# ---------------------------------------------------------------------------
# Pixel-level diversity & speed
# ---------------------------------------------------------------------------

def compute_pixel_diversity(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    device: torch.device,
    n_samples: int = 256,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Mean pairwise L2 + per-pixel std (overall, across whatever the
    ``sample_inputs`` distribution is)."""
    fakes = _generate_batches(generator, sample_inputs, n_samples, batch_size, device)
    flat = fakes.view(fakes.size(0), -1)
    dists = torch.cdist(flat, flat, p=2)
    mask = ~torch.eye(flat.size(0), dtype=torch.bool)
    return {
        "mean_pairwise_l2": float(dists[mask].mean().item()),
        "mean_pixel_std": float(flat.std(dim=0).mean().item()),
    }


def measure_generation_speed(
    generator: nn.Module,
    sample_inputs: SampleInputs,
    device: torch.device,
    n_samples: int = 1024,
    batch_size: int = 64,
    warmup: int = 1,
) -> float:
    """Return samples-per-second for ``generator``."""
    generator.eval()
    with torch.no_grad():
        for _ in range(warmup):
            inputs = _ensure_tuple(sample_inputs(min(batch_size, n_samples)))
            generator(*tuple(t.to(device) for t in inputs))

        start = time.time()
        remaining = n_samples
        while remaining > 0:
            n = min(batch_size, remaining)
            inputs = _ensure_tuple(sample_inputs(n))
            generator(*tuple(t.to(device) for t in inputs))
            remaining -= n
        elapsed = time.time() - start
    return n_samples / elapsed if elapsed > 0 else float("inf")


# ---------------------------------------------------------------------------
# Latent-space interpolation
# ---------------------------------------------------------------------------

def interpolate_points(p1: torch.Tensor, p2: torch.Tensor, n_steps: int = 12) -> torch.Tensor:
    """Linear interpolation between two latent vectors."""
    ratios = np.linspace(0.0, 1.0, num=n_steps)
    vectors = [(1.0 - r) * p1 + r * p2 for r in ratios]
    return torch.stack(vectors)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
