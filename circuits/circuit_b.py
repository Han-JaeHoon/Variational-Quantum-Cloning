# circuits/circuit_b.py

import pennylane as qml
import torch
import numpy as np

from .base import BaseCircuit


class CircuitB(BaseCircuit):
    """
    Variational quantum cloning circuit (Fig. 2b).

    - Trainable
    - 3 parameters per layer
    - Layer repetition configurable
    """

    def __init__(self, n_layers: int):

        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=3)
        self.qnode = self._build_qnode()

    # ---------------------------------
    # Interface properties
    # ---------------------------------

    @property
    def trainable(self):
        return True

    @property
    def n_params_per_layer(self):
        return 3

    # ---------------------------------
    # QNode construction
    # ---------------------------------

    def _build_qnode(self):

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, eta):

            # ----- Input state -----
            qml.Hadamard(wires=0)
            qml.RZ(eta, wires=0)

            # ----- Variational layers -----
            for l in range(self.n_layers):

                qml.RY(params[l, 0], wires=1)
                qml.CNOT(wires=[1, 2])

                qml.RY(params[l, 1], wires=2)
                qml.CNOT(wires=[2, 1])

                qml.RY(params[l, 2], wires=1)

            # ----- Cloning block -----
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[2, 0])

            return self._measure_fidelity(eta)

        return circuit

    # ---------------------------------
    # Fidelity measurement
    # ---------------------------------

    def _measure_fidelity(self, eta):

        psi = torch.stack([
            torch.tensor(1 / np.sqrt(2), dtype=torch.cdouble),
            torch.exp(1j * eta) / np.sqrt(2)
        ])

        projector = torch.outer(psi, torch.conj(psi))

        F_B = qml.expval(qml.Hermitian(projector, wires=1))
        F_E = qml.expval(qml.Hermitian(projector, wires=2))

        return F_B, F_E

    # ---------------------------------
    # Public API
    # ---------------------------------

    def fidelity(self, eta, params):
        return self.qnode(params, eta)