# models/variational_cloner.py

import torch
import torch.nn as nn


class VariationalCloner(nn.Module):
    """
    Wraps any BaseCircuit.

    - If circuit.trainable == True:
        → creates trainable parameters
    - If False:
        → no parameters created
    """

    def __init__(self, circuit):
        super().__init__()

        self.circuit = circuit

        if circuit.trainable:
            self.params = nn.Parameter(
                2 * torch.pi * torch.rand(
                    circuit.n_layers,
                    circuit.n_params_per_layer
                )
            )
        else:
            self.params = None

    def forward(self, eta):

        if self.circuit.trainable:
            F_B, F_E = self.circuit.fidelity(eta, self.params)
        else:
            F_B, F_E = self.circuit.fidelity(eta)

        cost = (
            (1 - F_B) ** 2
            + (1 - F_E) ** 2
            + (F_B - F_E) ** 2
        )

        return cost, F_B, F_E