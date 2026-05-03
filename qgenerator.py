"""Conditional patch-method quantum generator.

The generator is composed of ``n_generators`` sub-generators, all sharing the
same circuit architecture (defined in :mod:`qcircuit`). The conditional
extension is **purely classical** and lives entirely in this module: a
learnable embedding ``theta_class[c]`` produces a ``(n_qubits,)`` rotation
offset that is *added* to the latent vector ``z`` before it is encoded into
the circuit. Because consecutive ``RY`` rotations on the same qubit compose
as ``RY(a) RY(b) = RY(a + b)``, this is mathematically equivalent to
inserting per-class ``RY(theta_class[c, i])`` rotations right after the
state-embedding stage of the circuit -- so the circuit topology in
``qcircuit.py`` does not need to change.

Parameter budget for the default tutorial config (``5`` qubits, ``1``
ancilla, ``q_depth=6``, ``n_generators=4``, ``n_classes=10``):

* per sub-generator weights : ``q_depth * n_qubits = 30``
* total sub-generator weights: ``n_generators * 30 = 120``
* class embedding (shared)   : ``n_classes * n_qubits = 50``
* total trainable parameters : ``170``
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from dataset import N_CLASSES
from qcircuit import QCircuitConfig, make_partial_measure


class ConditionalPatchQuantumGenerator(nn.Module):
    """Class-conditional patch quantum generator.

    Parameters
    ----------
    n_generators:
        Number of sub-generators (``N_G``). Their outputs are concatenated to
        form a single ``patch_size * n_generators`` pixel image.
    n_classes:
        Number of conditioning classes (10 for the UCI handwritten digits).
    config:
        Circuit hyper-parameters. Defaults to the tutorial values
        (5 qubits, 1 ancilla, depth 6).
    q_delta:
        Spread of the uniform distribution used to initialise the parameters.
    cond_init_scale:
        Spread of the uniform distribution used to initialise the per-class
        rotation offsets. Kept small (``0.1`` rad) so the generator starts
        close to its unconditional behaviour and learns class structure
        gradually.
    partial_measure, device_name, diff_method:
        Same plumbing options as the unconditional generator. ``partial_measure``
        can be supplied to share a QNode across instances.
    """

    def __init__(
        self,
        n_generators: int,
        n_classes: int = N_CLASSES,
        config: Optional[QCircuitConfig] = None,
        q_delta: float = 1.0,
        cond_init_scale: float = 0.1,
        partial_measure: Optional[Callable] = None,
        device_name: str = "lightning.qubit",
        diff_method: str = "parameter-shift",
    ) -> None:
        super().__init__()

        self.config = config or QCircuitConfig()
        self.n_generators = n_generators
        self.n_classes = n_classes

        self.partial_measure = partial_measure or make_partial_measure(
            self.config, device_name=device_name, diff_method=diff_method
        )

        # Per-sub-generator parameterised weights.
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(
                    q_delta * torch.rand(self.config.n_weights),
                    requires_grad=True,
                )
                for _ in range(n_generators)
            ]
        )

        # Shared class-conditional rotation offsets (one row per class).
        # Small symmetric init keeps the generator close to its unconditional
        # behaviour at the start of training.
        self.class_emb = nn.Embedding(n_classes, self.config.n_qubits)
        nn.init.uniform_(
            self.class_emb.weight, -cond_init_scale, cond_init_scale
        )

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def output_size(self) -> int:
        return self.patch_size * self.n_generators

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate a batch of images conditioned on class labels.

        Parameters
        ----------
        z:
            Latent batch of shape ``(batch_size, n_qubits)`` with values
            sampled uniformly in ``[0, pi/2)``.
        y:
            Long tensor of class indices, shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Image batch of shape ``(batch_size, output_size)``.
        """
        if y.dtype != torch.long:
            y = y.long()

        device = z.device
        patch_size = self.patch_size

        # Apply the per-class rotation offsets to the latent vector. This is
        # equivalent to a class-conditional RY rotation on each qubit right
        # after the state-embedding stage.
        z_cond = z + self.class_emb(y)

        images = torch.empty(z.size(0), 0, device=device)
        for params in self.q_params:
            patches = torch.empty(0, patch_size, device=device)
            for elem in z_cond:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), dim=1)

        return images


# Backwards-compatible alias.
PatchQuantumGenerator = ConditionalPatchQuantumGenerator
