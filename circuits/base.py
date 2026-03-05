# circuits/base.py

from abc import ABC, abstractmethod


class BaseCircuit(ABC):
    """
    Abstract base class for all quantum cloning circuits.

    All circuit implementations (B, C, D, MBQC, etc.) must:
        - define whether they are trainable
        - implement fidelity(eta, params=None)
    """

    @property
    @abstractmethod
    def trainable(self):
        """
        Returns:
            bool: True if the circuit has learnable parameters
        """
        pass

    @abstractmethod
    def fidelity(self, eta, params=None):
        """
        Compute fidelities (F_B, F_E) for given phase eta.

        Args:
            eta (torch.Tensor): phase parameter
            params (torch.Tensor or None): trainable parameters if applicable

        Returns:
            tuple: (F_B, F_E)
        """
        pass