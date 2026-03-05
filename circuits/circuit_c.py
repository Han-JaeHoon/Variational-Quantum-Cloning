# circuits/circuit_c.py

"""
CircuitC: Fixed (non-trainable) analytic quantum cloning circuit
(corresponds to Fig. 2(c) of the paper).

This circuit:

    • Has no trainable parameters
    • Uses predetermined rotation angles
    • Serves as a reference (analytic) cloning circuit
    • Is evaluated directly without optimization

It implements a fixed cloning transformation and allows
comparison with the variational CircuitB.
"""

import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

from circuits.base import BaseCircuit


class CircuitC(BaseCircuit):

    def __init__(self):
        """
        Initialize 3-qubit device and construct two QNodes:

            - fid_qnode   : returns fidelities only
            - state_qnode : returns full statevector (for inspection)

        No parameters are required for this circuit.
        """

        self.dev = qml.device("default.qubit", wires=3)

        # QNode used to compute fidelities (no gradients needed)
        self.fid_qnode = self._build_fidelity_qnode()

        # QNode used for full state inspection (testing only)
        self.state_qnode = self._build_state_qnode()

    # ------------------------------------------------------------------
    # Circuit property
    # ------------------------------------------------------------------

    @property
    def trainable(self):
        """
        This circuit contains no learnable parameters.

        Trainer will automatically skip optimization
        when this property returns False.
        """
        return False

    # ------------------------------------------------------------
    # Circuit blocks
    # ------------------------------------------------------------

    def _prepare_input(self, eta):
        """
        Prepare the phase-covariant input state:

            |ψ(η)> = (|0> + e^{iη}|1>) / sqrt(2)

        Implemented as:
            Hadamard → RZ(η)
        """
        qml.Hadamard(wires=0)
        qml.RZ(eta, wires=0)
        qml.Barrier()

    def _fixed_circuit(self):
        """
        Fixed analytic gate sequence.

        The rotation angles are predetermined (no training).
        These values correspond to the optimized structure
        presented in the paper.

        Acts on three wires:
            wire 0 : input qubit
            wire 1 : clone B
            wire 2 : clone E
        """

        qml.RZ(15*np.pi/8, wires=0)
        qml.RY(np.pi/4, wires=1)
        qml.RX(3*np.pi/16, wires=2)

        qml.CZ(wires=[0,1])
        qml.RZ(14*np.pi/8, wires=2)

        qml.RX(11*np.pi/16, wires=1)
        qml.RY(13*np.pi/8, wires=2)

        qml.CZ(wires=[0,2])

        qml.RX(3*np.pi/2, wires=0)
        qml.RZ(np.pi, wires=1)
        qml.RY(93*np.pi/16, wires=2)

        qml.RZ(np.pi, wires=0)
        qml.RZ(3*np.pi/4, wires=2)

        qml.RX(13*np.pi/8, wires=2)

        qml.CZ(wires=[0,2])
        qml.CZ(wires=[1,2])

        qml.RY(15*np.pi/8, wires=0)
        qml.RY(3*np.pi/2, wires=1)

        qml.CZ(wires=[0,1])

    # ------------------------------------------------------------
    # QNodes
    # ------------------------------------------------------------

    def _build_fidelity_qnode(self):
        """
        Construct QNode that returns cloning fidelities.

        Since the circuit is fixed, no gradient method is required.
        """

        @qml.qnode(self.dev, interface="torch")
        def circuit(eta):

            # Prepare input state
            self._prepare_input(eta)

            # Apply fixed cloning transformation
            self._fixed_circuit()

            # Construct projector |ψ><ψ|
            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            projector = torch.outer(psi, torch.conj(psi))

            # IMPORTANT:
            # Wire assignment differs from CircuitB.
            #
            # In CircuitC:
            #   wire 0 → clone B
            #   wire 1 → clone E
            #
            # Fidelity is computed as:
            #   F = <ψ| ρ_clone |ψ>
            F_B = qml.expval(qml.Hermitian(projector, wires=0))
            F_E = qml.expval(qml.Hermitian(projector, wires=1))

            return F_B, F_E

        return circuit

    def _build_state_qnode(self):
        """
        Construct QNode that returns the full 3-qubit statevector.

        Used only for post-training inspection.
        Not used during optimization.
        """

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
        """
        Plot the full fixed cloning circuit structure.
        Uses dummy eta for visualization only.
        """

        # dummy_eta = torch.tensor(0.0)
        dummy_eta = torch.tensor(0.0, requires_grad=False)

        fig, ax = qml.draw_mpl(self.fid_qnode)(dummy_eta)

        ax.set_title("CircuitC Structure")
        # fig.show() # Use plt.show() instead for better control in Jupyter
        plt.show()

        print(qml.draw(self.fid_qnode)(dummy_eta))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def fidelity(self, eta, params=None):
        """
        Return (F_B, F_E) for a given phase η.

        'params' argument is ignored since the circuit is fixed.
        """
        return self.fid_qnode(eta)

    def analyze_states(self, etas, params=None):
        """
        Perform detailed inspection over fixed test phases.

        For each η:
            1. Construct 2D input state |ψ>
            2. Compute full 3-qubit state
            3. Extract reduced density matrices:
                   ρ_B (wire 0)
                   ρ_E (wire 1)
            4. Compute fidelities
            5. Print all quantities

        This method is for analysis only and does not affect training.
        """

        print("\n===== TEST ANALYSIS (Circuit C) =====\n")

        for i, eta in enumerate(etas):

            # ----- Input state (2D Hilbert space) -----
            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            # Pure input density matrix
            rho_input = torch.outer(psi, torch.conj(psi))

            # ----- Full 3-qubit state (8D) -----
            full_state = self.state_qnode(eta)
            rho_full = torch.outer(full_state, torch.conj(full_state))

            # Reduced density matrices of clones
            rho_B = qml.math.reduce_dm(rho_full, indices=[0])
            rho_E = qml.math.reduce_dm(rho_full, indices=[1])

            F_B, F_E = self.fidelity(eta)

            print(f"--- State {i+1} ---")
            print("eta:", eta.item())
            print("\nInput state vector (2D):\n", psi.detach().numpy())
            print("\nInput density matrix:\n", rho_input.detach().numpy())
            print("\nrho_B:\n", rho_B.detach().numpy())
            print("\nrho_E:\n", rho_E.detach().numpy())
            print("\nFidelity B:", F_B.item())
            print("Fidelity E:", F_E.item())
            print("-"*50)