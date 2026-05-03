"""Quantum circuit used by every sub-generator of the patch quantum GAN.

The circuit is described in section "Implementing the Generator" of the
PennyLane *Quantum GANs* tutorial. Each call to :func:`partial_measure`
performs the four stages described in the tutorial:

1. **State Embedding** -- a uniform latent vector ``z`` is encoded with one
   ``RY`` rotation per qubit.
2. **Parameterised Layers** -- ``q_depth`` blocks of trainable ``RY``
   rotations followed by a ladder of ``CZ`` entanglers.
3. **Non-Linear Transform** -- ancillary qubits are projected onto
   ``|0...0>`` by post-selection of the corresponding probability slice and
   re-normalisation.
4. **Post Processing** -- the resulting probability vector is divided by its
   maximum, mapping the patch into the ``[0, 1]`` pixel range.

The defaults below match the original tutorial exactly (``5`` qubits, ``1``
ancilla, depth ``6``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pennylane as qml
import torch


@dataclass(frozen=True)
class QCircuitConfig:
    """Hyper-parameters describing the patch-method circuit."""

    n_qubits: int = 5
    """Total number of qubits per sub-generator (``N`` in the tutorial)."""

    n_a_qubits: int = 1
    """Number of ancillary qubits used for the non-linear transform."""

    q_depth: int = 6
    """Number of repeated parameterised + entangling layers (``D``)."""

    @property
    def patch_size(self) -> int:
        """Number of pixels produced by a single sub-generator."""
        return 2 ** (self.n_qubits - self.n_a_qubits)

    @property
    def n_weights(self) -> int:
        """Number of trainable parameters in one sub-generator."""
        return self.q_depth * self.n_qubits


def make_quantum_circuit(
    config: QCircuitConfig | None = None,
    device_name: str = "lightning.qubit",
    diff_method: str = "parameter-shift",
) -> Callable:
    """Build the parameterised QNode used by every sub-generator.

    The returned callable has the signature ``quantum_circuit(noise, weights)``
    where ``noise`` is a ``(n_qubits,)`` latent vector in ``[0, pi/2)`` and
    ``weights`` is a flat tensor of length ``q_depth * n_qubits``.
    """

    cfg = config or QCircuitConfig()
    dev = qml.device(device_name, wires=cfg.n_qubits)

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(cfg.q_depth, cfg.n_qubits)

        # 1) State embedding: encode the latent vector with RY rotations.
        for i in range(cfg.n_qubits):
            qml.RY(noise[i], wires=i)

        # 2) Parameterised layers + CZ entanglers, repeated `q_depth` times.
        for d in range(cfg.q_depth):
            for w in range(cfg.n_qubits):
                qml.RY(weights[d][w], wires=w)
            for w in range(cfg.n_qubits - 1):
                qml.CZ(wires=[w, w + 1])

        return qml.probs(wires=list(range(cfg.n_qubits)))

    return quantum_circuit


def make_partial_measure(
    config: QCircuitConfig | None = None,
    device_name: str = "lightning.qubit",
    diff_method: str = "parameter-shift",
) -> Callable:
    """Build the patch generator's non-linear post-processing pipeline.

    See https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
    for background on how the partial measurement is realised in PennyLane.
    """

    cfg = config or QCircuitConfig()
    quantum_circuit = make_quantum_circuit(cfg, device_name, diff_method)

    def partial_measure(noise: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 3) Non-linear transform: keep only the probabilities that correspond
        #    to the ancilla being measured in |0...0>, then re-normalise.
        probs = quantum_circuit(noise, weights)
        probs_given_zero = probs[: cfg.patch_size]
        probs_given_zero = probs_given_zero / torch.sum(probs)

        # 4) Post-processing: divide by the max so the patch lives in [0, 1].
        return probs_given_zero / torch.max(probs_given_zero)

    return partial_measure
