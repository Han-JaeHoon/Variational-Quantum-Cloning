"""
main.py

Command-line controlled entry point for
Variational Quantum Cloning experiment.

Usage examples:

    python main.py --circuit B --layers 2
    python main.py --circuit C
    python main.py --circuit D
"""

import argparse
import torch
import numpy as np

from circuits.circuit_b import CircuitB
from circuits.circuit_c import CircuitC
from circuits.circuit_d import CircuitD
from models.variational_cloner import VariationalCloner
from data.phase_covariant_dataset import PhaseCovariantDataset
from trainer.trainer import Trainer


# ------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--circuit",
    type=str,
    required=True,
    choices=["B", "C", "D"],
    help="Choose cloning circuit: B (variational), C, or D"
)

parser.add_argument(
    "--layers",
    type=int,
    default=1,
    help="Number of ansatz layers (only used for CircuitB)"
)

args = parser.parse_args()


# ------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------

dataset = PhaseCovariantDataset()


# ------------------------------------------------------------
# 3. Circuit selection
# ------------------------------------------------------------

if args.circuit == "B":
    circuit = CircuitB(n_layers=args.layers)
elif args.circuit == "C":
    circuit = CircuitC()
elif args.circuit == "D":
    circuit = CircuitD()


print(f"\nSelected circuit: {args.circuit}")


# ------------------------------------------------------------
# 4. Model
# ------------------------------------------------------------

model = VariationalCloner(circuit)


# ------------------------------------------------------------
# 5. Trainer
# ------------------------------------------------------------

trainer = Trainer(model, dataset)


# ------------------------------------------------------------
# 6. Plot structure (once)
# ------------------------------------------------------------

circuit.plot_circuit()


# ------------------------------------------------------------
# 7. Training (if trainable)
# ------------------------------------------------------------

trainer.train()


# ------------------------------------------------------------
# 8. Fixed 10-point evaluation
# ------------------------------------------------------------

test_etas = torch.linspace(0, 2*np.pi, 10)

if circuit.trainable:
    circuit.analyze_states(test_etas, model.params)
else:
    circuit.analyze_states(test_etas)