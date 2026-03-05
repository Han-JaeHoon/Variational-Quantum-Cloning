# models/variational_cloner.py

"""
VariationalCloner

This module wraps a quantum cloning circuit into a PyTorch nn.Module.

Responsibilities:

    • Hold trainable parameters (if the circuit is variational)
    • Call the circuit to compute fidelities
    • Compute the cloning cost function
    • Provide a standard forward() interface for training

This allows the quantum circuit to integrate seamlessly
with PyTorch optimizers (e.g., Adam).
"""

import torch
import torch.nn as nn


class VariationalCloner(nn.Module):

    def __init__(self, circuit):
        """
        Args:
            circuit (BaseCircuit):
                Either CircuitB (trainable) or CircuitC (fixed).

        If the circuit is trainable:
            - Initialize parameters as nn.Parameter
            - Parameters shape: (n_layers, n_params_per_layer)

        If not trainable:
            - No parameters are created
        """

        super().__init__()

        self.circuit = circuit

        # ------------------------------------------------------------
        # Parameter initialization (only if circuit is trainable)
        # ------------------------------------------------------------
        if circuit.trainable:

            # Random initialization in [0, 2π)
            self.params = nn.Parameter(
                2 * torch.pi * torch.rand(
                    circuit.n_layers,
                    circuit.n_params_per_layer
                )
            )
        else:
            # Fixed circuit: no learnable parameters
            self.params = None

    def forward(self, eta):
        """
        Forward pass for a single phase value η.

        Steps:
            1. Call circuit to compute fidelities
            2. Compute cloning cost function
            3. Return (cost, F_B, F_E)

        Cost function (from paper):

            C = (1 - F_B)^2
              + (1 - F_E)^2
              + (F_B - F_E)^2

        The third term enforces symmetry between clones.
        """

        # ------------------------------------------------------------
        # Compute fidelities from quantum circuit
        # ------------------------------------------------------------
        if self.circuit.trainable:
            F_B, F_E = self.circuit.fidelity(eta, self.params)
        else:
            F_B, F_E = self.circuit.fidelity(eta)

        # ------------------------------------------------------------
        # Squared local cloning cost
        # ------------------------------------------------------------
        cost = (
            (1 - F_B)**2
            + (1 - F_E)**2
            + (F_B - F_E)**2
        )

        return cost, F_B, F_E