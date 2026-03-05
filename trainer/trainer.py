# trainer/trainer.py

import torch

class Trainer:

    def __init__(self, model, dataset, lr=0.05):
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, epochs=200, batch_size=10):

        for epoch in range(epochs):

            batch_idx = torch.randperm(len(self.dataset.train))[:batch_size]
            batch = self.dataset.train[batch_idx]

            total_loss = 0
            total_FB = 0
            total_FE = 0

            for eta in batch:
                loss, FB, FE = self.model(eta)
                total_loss += loss
                total_FB += FB
                total_FE += FE

            total_loss /= batch_size
            total_FB /= batch_size
            total_FE /= batch_size

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}")
                print(f"  Loss      : {total_loss.item():.6f}")
                print(f"  Fidelity B: {total_FB.item():.6f}")
                print(f"  Fidelity E: {total_FE.item():.6f}")
                print("")