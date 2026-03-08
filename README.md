# Variational Quantum Cloning

Implementation of variational and analytic quantum cloning circuits   using PennyLane + PyTorch. \
(Ref : https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.042604)

This project reproduces and compares three cloning circuit architectures:

- **Circuit B** — Variational (trainable)
- **Circuit C** — Fixed analytic circuit
- **Circuit D** — Alternative fixed analytic circuit

The goal is to compare cloning fidelities for phase-covariant input states.

---

# Installation
Clone this repository on your computer:
```bash
git clone https://github.com/Han-JaeHoon/Variational-Quantum-Cloning.git
cd ./Variational-Quantum-Cloning
```


Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

# 🚀 Running Experiments

## 1️⃣ Variational Circuit (B)

```bash
python main.py --circuit B --layers 1
```

Optional:
```bash
--layers N
```

Controls number of ansatz layers.

---

## 2️⃣ Fixed Circuit C

```bash
python main.py --circuit C
```

No training is performed.

---

## 3️⃣ Fixed Circuit D

```bash
python main.py --circuit D
```


---

# 📂 Project Structure
```
.
├── main.py
│
├── circuits/
│ ├── base.py
│ ├── circuit_b.py
│ ├── circuit_c.py
│ └── circuit_d.py
│
├── models/
│ └── variational_cloner.py
│
├── data/
│ └── phase_covariant_dataset.py
│
└── trainer/
└── trainer.py
```

---

# 🧩 File Descriptions

## `main.py`

Entry point for experiments.

- Parses command-line arguments
- Selects circuit (B, C, or D)
- Builds model
- Trains if needed
- Runs fixed 10-point evaluation
- Plots circuit structure

---

## `circuits/base.py`

Abstract base class defining the common interface:

- `trainable`
- `fidelity()`
- `analyze_states()`

All circuits inherit from this.

---

## `circuits/circuit_b.py`

Variational cloning circuit.

Features:
- Trainable ansatz
- Configurable number of layers
- Parameter-shift gradients
- Fidelity-based cost function

Each ansatz layer contains 3 parameters.

---

## `circuits/circuit_c.py`

Fixed analytic cloning circuit (Fig. 2(c)).

- No trainable parameters
- Direct fidelity evaluation
- Used as reference baseline

---

## `circuits/circuit_d.py`

Alternative fixed analytic cloning circuit (Fig. 2(d)).

- Same evaluation pipeline as Circuit C
- Different internal gate structure

---

## `models/variational_cloner.py`

PyTorch wrapper around circuit.

- Initializes parameters (if trainable)
- Defines cloning cost function:

$$
\mathcal{L} =
(1 - F_B)^2 +
(1 - F_E)^2 +
(F_B - F_E)^2
$$

Encourages:
- High fidelity
- Symmetric cloning

---

## `data/phase_covariant_dataset.py`

Generates random phases:

$$
\eta \in [0, 2\pi)
$$

Split into:
- Training set
- Test set

NOTE:
The fixed 10-point evaluation at the end of `main.py`
is NOT part of training.

---

## `trainer/trainer.py`

Handles training loop:

- Mini-batch sampling
- Gradient update (Adam)
- Loss tracking
- Fidelity tracking
- Training curve plotting

Skipped automatically for fixed circuits.

---

# 📊 Output

When running:

- Circuit structure is plotted
- Training loss + fidelities plotted (if trainable)
- Final 10-point evaluation printed:
  - Input state vector
  - Input density matrix
  - Reduced density matrices ρ_B, ρ_E
  - Fidelities

---

# 🧠 Input State

All circuits clone the phase-covariant state:

$$
|\psi(\eta)\rangle =
\frac{|0\rangle + e^{i\eta}|1\rangle}{\sqrt{2}}
$$

Prepared via:

```
Hadamard → RZ(η)
```

---

# 📈 Research Notes

- Circuit B: Expressivity depends on number of layers.
- Circuit C/D: Analytic constructions.
- Fidelity computed as:

$$ F = \langle \psi | \rho_{clone} | \psi \rangle $$


using projector measurement.

---
