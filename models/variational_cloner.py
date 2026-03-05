# models/variational_cloner.py

import torch
import torch.nn as nn

class VariationalCloner(nn.Module):

    def __init__(self, circuit_class, n_layers):
        super().__init__()

        self.circuit = circuit_class(n_layers)
        self.params = nn.Parameter(
            2 * torch.pi * torch.rand(n_layers, 3)
        )

    def forward(self, eta):

        F_B, F_E = self.circuit.qnode(self.params, eta)

        cost = (
            (1-F_B)**2
            + (1-F_E)**2
            + (F_B - F_E)**2
        )

        return cost, F_B, F_E