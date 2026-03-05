# trainer/trainer.py

import torch


class Trainer:

    def __init__(self, model, dataset, lr=0.05):
        self.model = model
        self.dataset = dataset

        if model.circuit.trainable:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr
            )
        else:
            self.optimizer = None

    def train(self, epochs=200, batch_size=10):

        print("\n========== TRAINING START ==========\n")

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

            if self.model.circuit.trainable:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}")
                print(f"  Loss       : {total_loss.item():.6f}")
                print(f"  Fidelity B : {total_FB.item():.6f}")
                print(f"  Fidelity E : {total_FE.item():.6f}")
                print("")

        print("\n========== TRAINING COMPLETE ==========\n")

    def evaluate(self):

        F_B_list = []
        F_E_list = []

        for eta in self.dataset.test:
            _, FB, FE = self.model(eta)
            F_B_list.append(FB.item())
            F_E_list.append(FE.item())

        print("Test Results")
        print("  Avg Fidelity B :", sum(F_B_list) / len(F_B_list))
        print("  Avg Fidelity E :", sum(F_E_list) / len(F_E_list))