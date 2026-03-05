# circuits/circuit_b.py

"""
CircuitB: Variational quantum cloning circuit (corresponds to Fig. 2(b)).

- Trainable circuit
- Supports configurable number of ansatz layers
- Uses parameter-shift rule for gradient computation
- Returns fidelities (F_B, F_E)

Structure:
    Input preparation
        ↓
    Variational ansatz (L layers)
        ↓
    Fixed cloning interaction block
        ↓
    Fidelity measurement
"""

import pennylane as qml
import torch
import numpy as np

from circuits.base import BaseCircuit

class CircuitB(BaseCircuit):

    def __init__(self, n_layers):
        """
        Args:
            n_layers (int): Number of repeated ansatz layers.
                            Each layer has 3 trainable parameters.
        """

        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=3)

        # QNode used during training (returns fidelities only)
        self.fid_qnode = self._build_fidelity_qnode()

        # QNode used for post-training analysis (returns full state)
        self.state_qnode = self._build_state_qnode()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trainable(self):
        """Indicates this circuit contains learnable parameters."""
        return True

    @property
    def n_params_per_layer(self):
        """Number of trainable parameters per ansatz layer."""
        return 3

    # ------------------------------------------------------------------
    # QNode builders
    # ------------------------------------------------------------------

    def _build_fidelity_qnode(self):
        """
        Build training QNode.

        Uses parameter-shift rule and returns only expectation values
        (no state output, to remain gradient-compatible).
        """

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, eta):

            self._prepare_input(eta)
            self._ansatz(params)
            self._cloning_block()

            return self._measure_fidelity(eta)

        return circuit

    def _build_state_qnode(self):
        """
        Build analysis QNode.

        Returns full statevector (used only for testing/inspection).
        """

        @qml.qnode(self.dev, interface="torch")
        def circuit(params, eta):

            self._prepare_input(eta)
            self._ansatz(params)
            self._cloning_block()

            return qml.state()

        return circuit

    # ------------------------------------------------------------------
    # Circuit building blocks
    # ------------------------------------------------------------------

    def _prepare_input(self, eta):
        """
        Prepare phase-covariant input state:

            |ψ(η)> = (|0> + e^{iη}|1>) / sqrt(2)

        Implemented as:
            H → RZ(η)
        """
        qml.Hadamard(wires=0)
        qml.RZ(eta, wires=0)
        qml.Barrier()

    def _ansatz(self, params):
        """
        Variational ansatz.

        Each layer:
            RY → CNOT → RY → CNOT → RY

        Acts on wires 1 and 2.
        """

        for l in range(self.n_layers):

            qml.RY(params[l, 0], wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RY(params[l, 1], wires=2)
            qml.CNOT(wires=[2, 1])

            qml.RY(params[l, 2], wires=1)

    def _cloning_block(self):
        """
        Fixed cloning interaction block.

        Entangles input qubit (wire 0) with clone qubits.
        """

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 0])
        qml.CNOT(wires=[2, 0])

    def _measure_fidelity(self, eta):
        """
        Compute fidelities:

            F_B = <ψ|ρ_B|ψ>
            F_E = <ψ|ρ_E|ψ>

        Implemented via expectation value of projector |ψ><ψ|.
        """

        psi = torch.stack([
            torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
            torch.exp(1j*eta)/np.sqrt(2)
        ])

        projector = torch.outer(psi, torch.conj(psi))

        F_B = qml.expval(qml.Hermitian(projector, wires=1))
        F_E = qml.expval(qml.Hermitian(projector, wires=2))

        return F_B, F_E

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_circuit(self):
        """
        Plot the full quantum circuit structure (one example instance).

        Uses:
            - Dummy parameters
            - Dummy eta
        This is for structural verification only.
        """

        # Dummy inputs (just for visualization)
        dummy_params = torch.zeros(
            self.n_layers,
            self.n_params_per_layer
        )
        dummy_eta = torch.tensor(0.0)

        # Use PennyLane's matplotlib drawer
        fig, ax = qml.draw_mpl(self.fid_qnode)(
            dummy_params,
            dummy_eta
        )

        ax.set_title("CircuitB Structure")
        fig.show()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fidelity(self, eta, params):
        """
        Return (F_B, F_E) for given phase and parameters.
        Used during training.
        """
        return self.fid_qnode(params, eta)

    def analyze_states(self, etas, params):
        """
        Post-training inspection over fixed test phases.

        For each η:
            - Print input state (2D)
            - Print input density matrix
            - Print reduced density matrices ρ_B, ρ_E
            - Print fidelities
        """

        print("\n===== TEST ANALYSIS (Circuit B) =====\n")

        for i, eta in enumerate(etas):

            # ----- Input state (2D) -----
            psi = torch.stack([
                torch.tensor(1/np.sqrt(2), dtype=torch.cdouble),
                torch.exp(1j*eta)/np.sqrt(2)
            ])

            rho_input = torch.outer(psi, torch.conj(psi))

            # ----- Full system state (3 qubits) -----
            full_state = self.state_qnode(params, eta)
            rho_full = torch.outer(full_state, torch.conj(full_state))

            # Reduced density matrices
            rho_B = qml.math.reduce_dm(rho_full, indices=[1])
            rho_E = qml.math.reduce_dm(rho_full, indices=[2])

            F_B, F_E = self.fidelity(eta, params)

            print(f"--- State {i+1} ---")
            print("eta:", eta.item())
            print("\nInput state vector (2D):\n", psi.detach().numpy())
            print("\nInput density matrix:\n", rho_input.detach().numpy())
            print("\nrho_B:\n", rho_B.detach().numpy())
            print("\nrho_E:\n", rho_E.detach().numpy())
            print("\nFidelity B:", F_B.item())
            print("Fidelity E:", F_E.item())
            print("-"*50)