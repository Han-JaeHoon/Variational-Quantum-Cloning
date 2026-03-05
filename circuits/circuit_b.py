# circuits/circuit_b.py

from .base import BaseCircuit
import pennylane as qml

class CircuitB(BaseCircuit):

    def _ansatz(self, params):

        for l in range(self.n_layers):

            qml.RY(params[l,0], wires=1)
            qml.CNOT(wires=[1,2])

            qml.RY(params[l,1], wires=2)
            qml.CNOT(wires=[2,1])

            qml.RY(params[l,2], wires=1)