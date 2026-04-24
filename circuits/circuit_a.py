# circuits/circuit_a.py

"""
CircuitA: Optimized Variational Quantum Cloning Circuit.

Modifications from original:
- Removed final Hadamard(wire=2)
- Removed final SWAP(0,2)
- Clone E is now on wire 0
- Clone B is now on wire 1

Training performance unchanged.
"""

import pennylane as qml
import torch
import numpy as np

from circuits.base import BaseCircuit


class CircuitA(BaseCircuit):

    # ==============================================================
    # Initialization
    # ==============================================================

    def __init__(self, n_layers):

        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=3)

        self.fid_qnode = self._build_fidelity_qnode()
        self.state_qnode = self._build_state_qnode()

    # ==============================================================
    # Properties
    # ==============================================================

    @property
    def trainable(self):
        return True

    @property
    def n_params_per_layer(self):
        return 3

    # ==============================================================
    # QNode builders
    # ==============================================================

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

    # ==============================================================
    # Circuit blocks
    # ==============================================================

    def _prepare_input(self, eta):

        # |ψ(η)> on wire 0
        qml.Hadamard(wires=0)
        qml.RZ(eta, wires=0)

        # Initialize blank clones
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.Barrier()

    # --------------------------------------------------------------

    def _ansatz(self, params):

        for l in range(self.n_layers):
            self._ansatz_layer(params[l])

    # --------------------------------------------------------------

    def _ansatz_layer(self, theta):

        # ----- Local rotation (wire 1) -----
        qml.RZ(theta[0], wires=1)
        qml.Hadamard(wires=1)
        qml.CZ(wires=[1, 2])

        # ----- Rotations (wire 2) -----
        qml.RZ(np.pi / 2, wires=1)
        qml.Hadamard(wires=1)

        qml.RZ(np.pi / 2, wires=2)
        qml.Hadamard(wires=2)

        qml.RZ(theta[1], wires=2)
        qml.Hadamard(wires=2)
        qml.RZ(-np.pi / 2, wires=2)
        qml.Hadamard(wires=2)

        qml.CZ(wires=[2, 1])

        # ----- Entangle with input -----
        qml.RZ(-np.pi / 2, wires=1)
        qml.Hadamard(wires=1)
        qml.RZ(-theta[2], wires=1)
        qml.Hadamard(wires=1)

        qml.CZ(wires=[0, 1])

        # ----- Symmetrization block -----
        qml.Hadamard(wires=0)

        qml.RZ(np.pi / 2, wires=1)
        qml.Hadamard(wires=1)

        qml.CZ(wires=[0, 2])

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=2)

        qml.CZ(wires=[1, 2])

        # ❌ Removed:
        # qml.Hadamard(wires=2)
        # qml.SWAP(wires=[0, 2])

    # ==============================================================
    # Fidelity measurement
    # ==============================================================

    def _target_projector(self, eta):

        psi = torch.stack([
            torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
            torch.exp(1j * eta) / np.sqrt(2)
        ])

        return torch.outer(psi, torch.conj(psi))

    def _measure_fidelity(self, eta):

        projector = self._target_projector(eta)

        # 🔄 Clone E → wire 0
        # 🔄 Clone B → wire 1
        F_B = qml.expval(qml.Hermitian(projector, wires=1))
        F_E = qml.expval(qml.Hermitian(projector, wires=0))

        return F_B, F_E

    # ==============================================================
    # Visualization
    # ==============================================================

    def plot_circuit(self):

        dummy_params = torch.zeros(
            self.n_layers,
            self.n_params_per_layer
        )

        dummy_eta = torch.tensor(0.0)

        fig, ax = qml.draw_mpl(self.fid_qnode)(
            dummy_params,
            dummy_eta
        )

        ax.set_title("CircuitA (Optimized, No SWAP)")
        fig.show()

    # ==============================================================
    # Public API
    # ==============================================================

    def fidelity(self, eta, params):
        return self.fid_qnode(params, eta)

    # ==============================================================
    # Analysis
    # ==============================================================

    def analyze_states(self, etas, params):

        print("\n===== TEST ANALYSIS (Circuit A - SWAP Removed) =====\n")

        for i, eta in enumerate(etas):

            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j * eta) / np.sqrt(2)
            ])

            rho_input = torch.outer(psi, torch.conj(psi))

            full_state = self.state_qnode(params, eta)
            rho_full = torch.outer(full_state, torch.conj(full_state))

            # 🔄 Clone E → wire 0
            # 🔄 Clone B → wire 1
            rho_E = qml.math.reduce_dm(rho_full, indices=[0])
            rho_B = qml.math.reduce_dm(rho_full, indices=[1])

            F_B, F_E = self.fidelity(eta, params)

            print(f"--- State {i+1} ---")
            print("eta:", eta.item())
            print("\nInput density matrix:\n", rho_input.detach().numpy())
            print("\nrho_E (wire 0):\n", rho_E.detach().numpy())
            print("\nrho_B (wire 1):\n", rho_B.detach().numpy())
            print("\nFidelity B:", F_B.item())
            print("Fidelity E:", F_E.item())
            print("-"*50)