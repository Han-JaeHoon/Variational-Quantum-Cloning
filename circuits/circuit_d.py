"""
CircuitD: Alternative fixed (non-trainable) analytic cloning circuit
(corresponds to Fig. 2(d) of the paper).

Structure identical to CircuitC,
but internal gate sequence differs.
"""

import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

from circuits.base import BaseCircuit


class CircuitD(BaseCircuit):

    def __init__(self):

        self.dev = qml.device("default.qubit", wires=3)

        self.fid_qnode = self._build_fidelity_qnode()
        self.state_qnode = self._build_state_qnode()

    # ------------------------------------------------------------
    # Circuit property
    # ------------------------------------------------------------

    @property
    def trainable(self):
        return False

    # ------------------------------------------------------------
    # Circuit blocks
    # ------------------------------------------------------------

    def _prepare_input(self, eta):
        qml.Hadamard(wires=0)
        qml.RZ(eta, wires=0)
        qml.Barrier()

    def _fixed_circuit(self):
        """
        Gate sequence from Fig. 2(d).
        """

        # Top wire (A)
        qml.RZ(np.pi/8, wires=0)

        # Middle wire (E)
        qml.RX(13*np.pi/8, wires=1)

        # Bottom wire (E*)
        qml.RY(12*np.pi/8, wires=2)
        qml.RX(5*np.pi/8, wires=2)
        qml.RZ(3*np.pi/2, wires=2)
        qml.RX(np.pi/2, wires=2)

        # First entangling block
        qml.CZ(wires=[0,1])

        # Middle wire (E)
        qml.RZ(np.pi, wires=1)
        qml.RY(9*np.pi/8, wires=1)
        qml.RX(np.pi/4, wires=1)

        # Second entangling block
        qml.CZ(wires=[0,1])

        # Second block
        qml.RX(3*np.pi/2, wires=0)
        qml.RZ(np.pi/8, wires=0)
        qml.RY(np.pi/2, wires=1)
        qml.RZ(11*np.pi/8, wires=1)

        # Third entangling block
        qml.CZ(wires=[1,2])

        # Middle wire (E)
        qml.RY(3*np.pi/2, wires=1)

        # Final entangling block
        qml.CZ(wires=[0,1])

        # Final rotations
        qml.RX(3*np.pi/2, wires=0)
        qml.RY(15*np.pi/8, wires=0)

    # ------------------------------------------------------------
    # QNodes
    # ------------------------------------------------------------

    def _build_fidelity_qnode(self):

        @qml.qnode(self.dev, interface="torch")
        def circuit(eta):

            self._prepare_input(eta)
            self._fixed_circuit()

            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            projector = torch.outer(psi, torch.conj(psi))

            # Same wire convention as CircuitC
            F_B = qml.expval(qml.Hermitian(projector, wires=0))
            F_E = qml.expval(qml.Hermitian(projector, wires=1))

            return F_B, F_E

        return circuit

    def _build_state_qnode(self):

        @qml.qnode(self.dev, interface="torch")
        def circuit(eta):

            self._prepare_input(eta)
            self._fixed_circuit()

            return qml.state()

        return circuit

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    def plot_circuit(self):

        dummy_eta = torch.tensor(0.0)

        fig, ax = qml.draw_mpl(self.fid_qnode)(dummy_eta)
        ax.set_title("CircuitD Structure")
        plt.show()

        print(qml.draw(self.fid_qnode)(dummy_eta))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def fidelity(self, eta, params=None):
        return self.fid_qnode(eta)

    def analyze_states(self, etas, params=None):

        print("\n===== TEST ANALYSIS (Circuit D) =====\n")

        for i, eta in enumerate(etas):

            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            rho_input = torch.outer(psi, torch.conj(psi))

            full_state = self.state_qnode(eta)
            rho_full = torch.outer(full_state, torch.conj(full_state))

            rho_B = qml.math.reduce_dm(rho_full, indices=[0])
            rho_E = qml.math.reduce_dm(rho_full, indices=[1])

            F_B, F_E = self.fidelity(eta)

            print(f"--- State {i+1} ---")
            print("eta:", eta.item())
            print("\nInput density matrix:\n", rho_input.detach().numpy())
            print("\nrho_B:\n", rho_B.detach().numpy())
            print("\nrho_E:\n", rho_E.detach().numpy())
            print("\nFidelity B:", F_B.item())
            print("Fidelity E:", F_E.item())
            print("-"*50)