# circuits/base.py

"""
Base interface for all quantum cloning circuits.

This abstract class defines the minimal contract that every
cloning circuit (e.g., CircuitB, CircuitC, CircuitD, MBQC version, etc.)
must follow.

The goal of this base class is to enforce a unified interface so that:

    - Trainer
    - Model wrapper
    - Evaluation utilities

can interact with any circuit in a consistent way,
regardless of whether the circuit is:

    • trainable (variational)
    • fixed (analytic)
    • gate-based
    • MBQC-based
"""

from abc import ABC, abstractmethod


class BaseCircuit(ABC):
    """
    Abstract base class for quantum cloning circuits.

    Any subclass must implement:

        1. trainable property
        2. fidelity(eta, params)
        3. analyze_states(etas, params)

    This ensures that all circuit types behave identically
    from the perspective of higher-level components
    (Model, Trainer, etc.).
    """

    # ------------------------------------------------------------------
    # Whether the circuit contains trainable parameters
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def trainable(self):
        """
        Indicates whether the circuit contains learnable parameters.

        Returns:
            bool:
                True  → variational circuit (e.g., CircuitB)
                False → fixed analytic circuit (e.g., CircuitC)

        This property is used by the Trainer to decide:
            - whether to perform gradient updates
            - whether to skip training entirely
        """
        pass

    # ------------------------------------------------------------------
    # Fidelity computation interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fidelity(self, eta, params=None):
        """
        Compute the cloning fidelities for a given input phase.

        Args:
            eta (torch.Tensor):
                Phase parameter defining the input state:
                    |ψ(η)> = (|0> + e^{iη}|1>) / sqrt(2)

            params (torch.Tensor or None):
                Trainable parameters of the circuit.
                - Required for trainable circuits.
                - Ignored for fixed circuits.

        Returns:
            tuple:
                (F_B, F_E)

                F_B → fidelity of Bob clone
                F_E → fidelity of Eve clone

        This method must NOT return full quantum states.
        It should return expectation values only,
        so that gradient computation (parameter-shift)
        remains compatible.
        """
        pass

    # ------------------------------------------------------------------
    # Post-training analysis interface
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze_states(self, etas, params=None):
        """
        Perform detailed state analysis on a set of test phases.

        This method is intended for post-training inspection only.
        It should:

            - Construct the input state |ψ(η)>
            - Compute full system state
            - Extract reduced density matrices (ρ_B, ρ_E)
            - Print:
                • input state vector (2D)
                • input density matrix (2x2)
                • ρ_B
                • ρ_E
                • fidelities

        Args:
            etas (torch.Tensor):
                A fixed set of test phase values.
                (Typically evenly spaced over [0, 2π])

            params (torch.Tensor or None):
                Circuit parameters if trainable.

        Important:
            This method is NOT used during training.
            It may internally compute full quantum states
            (e.g., via qml.state()), since gradient
            computation is not required here.
        """
        pass