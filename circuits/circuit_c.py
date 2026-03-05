# circuits/circuit_c.py

import pennylane as qml
import torch
import numpy as np

from .base import BaseCircuit


class CircuitC(BaseCircuit):
    """
    Fixed analytic circuit (Fig. 2c).

    - Not trainable
    - No parameters
    - Uses predefined rotation angles
    """

    def __init__(self):

        self.dev = qml.device("default.qubit", wires=3)
        self.qnode = self._build_qnode()

    # ---------------------------------
    # Interface properties
    # ---------------------------------

    @property
    def trainable(self):
        return False

    # ---------------------------------
    # QNode construction
    # ---------------------------------

    def _build_qnode(self):

        @qml.qnode(self.dev, interface="torch")
        def circuit(eta):

            # ----- Input -----
            qml.Hadamard(wires=0)
            qml.RZ(eta, wires=0)

            # ----- Fixed analytic gates -----
            # (그림 Fig.2(c) 기반)
            qml.RZ(15 * np.pi / 8, wires=0)
            qml.RY(np.pi / 4, wires=1)
            qml.RX(3 * np.pi / 16, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

            qml.RX(11 * np.pi / 16, wires=1)
            qml.RY(9 * np.pi / 16, wires=2)

            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 1])

            qml.RZ(15 * np.pi / 8, wires=0)
            qml.RY(3 * np.pi / 2, wires=1)

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

        # Circuit C는 wire 위치 다름 (그림 기준)
        F_B = qml.expval(qml.Hermitian(projector, wires=0))
        F_E = qml.expval(qml.Hermitian(projector, wires=1))

        return F_B, F_E

    # ---------------------------------
    # Public API
    # ---------------------------------

    def fidelity(self, eta, params=None):
        return self.qnode(eta)