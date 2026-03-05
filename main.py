"""
main.py

Entry point for the variational quantum cloning experiment.

Pipeline:

    1. Generate phase dataset
    2. Select cloning circuit (trainable or fixed)
    3. Wrap circuit with PyTorch model
    4. Train (if applicable)
    5. Perform fixed 10-point analysis

This file controls high-level experiment configuration.
"""

from circuits.circuit_b import CircuitB
from circuits.circuit_c import CircuitC
from circuits.circuit_d import CircuitD
from models.variational_cloner import VariationalCloner
from data.phase_covariant_dataset import PhaseCovariantDataset
from trainer.trainer import Trainer
import torch
import numpy as np


# ------------------------------------------------------------
# 1. Dataset generation
# ------------------------------------------------------------
# Randomly samples phase parameters η ∈ [0, 2π)
# Used only for training/validation (NOT the fixed test below)
dataset = PhaseCovariantDataset()


# ------------------------------------------------------------
# 2. Circuit selection
# ------------------------------------------------------------
# Choose which cloning circuit to run:
#
#   CircuitB → trainable variational circuit
#   CircuitC → fixed analytic circuit (no training)
#
# For CircuitB, you can control model capacity via n_layers.
#
circuit = CircuitB(n_layers=1)
# circuit = CircuitC()
# circuit = CircuitD()


# ------------------------------------------------------------
# 3. Wrap circuit as PyTorch model
# ------------------------------------------------------------
# This creates trainable parameters if the circuit is variational.
model = VariationalCloner(circuit)


# ------------------------------------------------------------
# 4. Trainer setup
# ------------------------------------------------------------
trainer = Trainer(model, dataset)


# ------------------------------------------------------------
# 5. Training phase
# ------------------------------------------------------------
# 회로 구조 확인 (한 번만 실행)
circuit.plot_circuit()
# If circuit is trainable:
#     → performs gradient descent
# If not trainable:
#     → automatically skipped
trainer.train()


# ------------------------------------------------------------
# 6. Fixed 10-point evaluation
# ------------------------------------------------------------
# These test phases are evenly spaced in [0, 2π].
# IMPORTANT:
#   - These are NOT part of training.
#   - They are used only for post-training inspection.
#
# This ensures reproducibility and fair comparison
# between CircuitB and CircuitC.
#
test_etas = torch.linspace(0, 2*np.pi, 10)


# Perform detailed state analysis
if circuit.trainable:
    # Variational circuit requires trained parameters
    circuit.analyze_states(test_etas, model.params)
else:
    # Fixed circuit requires no parameters
    circuit.analyze_states(test_etas)