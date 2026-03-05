# data/phase_covariant_dataset.py

"""
PhaseCovariantDataset

This dataset generates random phase parameters η
for phase-covariant quantum cloning experiments.

The input quantum states are defined as:

    |ψ(η)> = (|0> + e^{iη}|1>) / sqrt(2)

Instead of storing full quantum states, we only store
the phase parameter η, since the state preparation
is done inside the quantum circuit.

This class:
    • Randomly samples η ∈ [0, 2π)
    • Splits them into train/test sets
    • Does NOT include the fixed 10 test states used for final inspection
      (those are generated separately in main.py)
"""

import torch
import numpy as np


class PhaseCovariantDataset:

    def __init__(self, n_samples=30, train_ratio=0.8):
        """
        Args:
            n_samples (int):
                Total number of randomly generated phase values.

            train_ratio (float):
                Fraction of samples used for training.
                Remaining samples are used for validation/testing.

        Example:
            n_samples=30, train_ratio=0.8
                → 24 training phases
                → 6 test phases
        """

        # ------------------------------------------------------------
        # Randomly sample phase values η ∈ [0, 2π)
        # ------------------------------------------------------------
        etas = 2*np.pi*torch.rand(n_samples)

        # ------------------------------------------------------------
        # Train/Test split
        # ------------------------------------------------------------
        split = int(train_ratio*n_samples)

        # Training set (used during optimization)
        self.train = etas[:split]

        # Test set (used for evaluation after training)
        self.test = etas[split:]