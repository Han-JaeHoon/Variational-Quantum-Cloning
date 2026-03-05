# data/phase_covariant_dataset.py

import torch
import numpy as np

class PhaseCovariantDataset:

    def __init__(self, n_samples=30, train_ratio=0.8):

        etas = 2*np.pi*torch.rand(n_samples)

        split = int(train_ratio*n_samples)

        self.train = etas[:split]
        self.test = etas[split:]