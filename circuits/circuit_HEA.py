# circuits/circuit_a.py

"""
CircuitA: Hardware-efficient variational quantum cloning circuit.

Structure per layer:
    - RX, RZ on all qubits
    - Linear CNOT chain: 0→1, 1→2, ...

Trainable circuit compatible with VariationalCloner and Trainer.
"""

import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

from circuits.base import BaseCircuit


class CircuitA(BaseCircuit):

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.n_qubits = 3
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.fid_qnode = self._build_fidelity_qnode()
        self.state_qnode = self._build_state_qnode()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trainable(self):
        return True

    @property
    def n_params_per_layer(self):
        # RX and RZ for each qubit
        return 2 * self.n_qubits  # 6 for 3 qubits

    # ------------------------------------------------------------------
    # QNode builders
    # ------------------------------------------------------------------

    def _build_fidelity_qnode(self):

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, eta):

            self._prepare_input(eta)
            self._ansatz(params)

            return self._measure_fidelity(eta)

        return circuit

    def _build_state_qnode(self):

        @qml.qnode(self.dev, interface="torch")
        def circuit(params, eta):

            self._prepare_input(eta)
            self._ansatz(params)

            return qml.state()

        return circuit

    # ------------------------------------------------------------------
    # Circuit blocks
    # ------------------------------------------------------------------

    def _prepare_input(self, eta):
        # Prepare |ψ(η)> on qubit 0, others initialized in |0>
        qml.Hadamard(wires=0)
        qml.RZ(eta, wires=0)
        qml.Barrier()

    def _ansatz(self, params):
        """
        Hardware-efficient ansatz:
        For each layer:
            - RX, RZ on all qubits
            - CNOT chain: 0->1, 1->2
        """

        for l in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                qml.RX(params[l, 2*q], wires=q)
                qml.RZ(params[l, 2*q + 1], wires=q)

            # Linear entangling chain
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q+1])

    def _measure_fidelity(self, eta):
        psi = torch.stack([
            torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
            torch.exp(1j * eta) / np.sqrt(2)
        ])

        projector = torch.outer(psi, torch.conj(psi))

        # Clone qubits: wire 1 (B), wire 2 (E)
        F_B = qml.expval(qml.Hermitian(projector, wires=1))
        F_E = qml.expval(qml.Hermitian(projector, wires=2))

        return F_B, F_E

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_circuit(self):
        dummy_params = torch.zeros(self.n_layers, self.n_params_per_layer)
        dummy_eta = torch.tensor(0.0)

        fig, ax = qml.draw_mpl(self.fid_qnode)(dummy_params, dummy_eta)
        ax.set_title("CircuitA Structure (Hardware Efficient)")
        plt.show()

        print(qml.draw(self.fid_qnode)(dummy_params, dummy_eta))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fidelity(self, eta, params):
        return self.fid_qnode(params, eta)

    def analyze_states(self, etas, params):
        print("\n===== TEST ANALYSIS (Circuit A) =====\n")

        for i, eta in enumerate(etas):
            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j * eta) / np.sqrt(2)
            ])

            rho_input = torch.outer(psi, torch.conj(psi))

            full_state = self.state_qnode(params, eta)
            rho_full = torch.outer(full_state, torch.conj(full_state))

            rho_B = qml.math.reduce_dm(rho_full, indices=[1])
            rho_E = qml.math.reduce_dm(rho_full, indices=[2])

            F_B, F_E = self.fidelity(eta, params)

            print(f"--- State {i+1} ---")
            print("eta:", eta.item())
            print("\nInput density matrix:\n", rho_input.detach().numpy())
            print("\nrho_B:\n", rho_B.detach().numpy())
            print("\nrho_E:\n", rho_E.detach().numpy())
            print("\nFidelity B:", F_B.item())
            print("Fidelity E:", F_E.item())
            print("-" * 50)