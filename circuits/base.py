# circuits/base.py

import pennylane as qml
import torch
import numpy as np

class BaseCircuit:

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=3)
        self.qnode = self._build_qnode()

    def _build_qnode(self):

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, eta):

            # Input
            qml.Hadamard(wires=0)
            qml.RZ(eta, wires=0)

            self._ansatz(params)

            # Cloning block
            qml.CNOT(wires=[0,1])
            qml.CNOT(wires=[0,2])
            qml.CNOT(wires=[1,0])
            qml.CNOT(wires=[2,0])

            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            projector = torch.outer(psi, torch.conj(psi))

            F_B = qml.expval(qml.Hermitian(projector, wires=1))
            F_E = qml.expval(qml.Hermitian(projector, wires=2))

            return F_B, F_E

        return circuit

    def _ansatz(self, params):
        raise NotImplementedError