from circuits.circuit_b import CircuitB
from circuits.circuit_c import CircuitC
from models.variational_cloner import VariationalCloner
from data.phase_covariant_dataset import PhaseCovariantDataset
from trainer.trainer import Trainer

dataset = PhaseCovariantDataset()

# ----------- 선택 -----------
circuit = CircuitB(n_layers=1)
# circuit = CircuitC()
# ----------------------------

model = VariationalCloner(circuit)
trainer = Trainer(model, dataset)

trainer.train()
trainer.evaluate()