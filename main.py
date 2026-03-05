from circuits.circuit_b import CircuitB
from models.variational_cloner import VariationalCloner
from data.phase_covariant_dataset import PhaseCovariantDataset
from trainer.trainer import Trainer

dataset = PhaseCovariantDataset()
model = VariationalCloner(CircuitB, n_layers=3)
trainer = Trainer(model, dataset)

trainer.train()