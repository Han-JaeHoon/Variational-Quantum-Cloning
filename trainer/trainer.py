# trainer/trainer.py

"""
Trainer

Responsible for:

    • Running the training loop (if circuit is trainable)
    • Performing gradient updates
    • Logging loss and fidelities
    • Plotting training dynamics

This class is agnostic to the specific circuit type.
It relies only on:

    model.forward(eta) → (cost, F_B, F_E)
"""

import torch
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, model, dataset, lr=0.05):
        """
        Args:
            model   : VariationalCloner instance
            dataset : PhaseCovariantDataset instance
            lr      : Learning rate (default = 0.05)

        Initializes:
            - Optimizer (only if circuit is trainable)
            - History trackers for visualization
        """

        self.model = model
        self.dataset = dataset

        # Store training curves
        self.loss_history = []
        self.fidB_history = []
        self.fidE_history = []

        # Initialize optimizer only if the circuit has trainable parameters
        if model.circuit.trainable:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = None

    def train(self, epochs=200, batch_size=10):
        """
        Run training loop.

        For each epoch:
            1. Sample random batch of phase values η
            2. Compute cost and fidelities
            3. Backpropagate gradients
            4. Update parameters
            5. Log statistics

        If the circuit is not trainable (e.g., CircuitC),
        training is skipped automatically.
        """

        # Skip training for fixed circuits
        if not self.model.circuit.trainable:
            print("Circuit not trainable. Skipping training.")
            return

        for epoch in range(epochs):

            # ------------------------------------------------------------
            # Random mini-batch sampling from training set
            # ------------------------------------------------------------
            batch_idx = torch.randperm(len(self.dataset.train))[:batch_size]
            batch = self.dataset.train[batch_idx]

            total_loss = 0
            total_FB = 0
            total_FE = 0

            # ------------------------------------------------------------
            # Forward pass for each η in batch
            # ------------------------------------------------------------
            for eta in batch:
                loss, FB, FE = self.model(eta)

                total_loss += loss
                total_FB += FB
                total_FE += FE

            # Average over batch
            total_loss /= batch_size
            total_FB /= batch_size
            total_FE /= batch_size

            # ------------------------------------------------------------
            # Backpropagation and parameter update
            # ------------------------------------------------------------
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # ------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------
            self.loss_history.append(total_loss.item())
            self.fidB_history.append(total_FB.item())
            self.fidE_history.append(total_FE.item())

            # Print progress every 20 epochs
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}")
                print(f"  Loss       : {total_loss.item():.6f}")
                print(f"  Fidelity B : {total_FB.item():.6f}")
                print(f"  Fidelity E : {total_FE.item():.6f}")
                print()

        # Plot training curves after completion
        self._plot_training()

    def _plot_training(self):
        """
        Plot training dynamics:

            - Loss curve
            - Fidelity B curve
            - Fidelity E curve

        This helps visualize:
            • Convergence behavior
            • Symmetry between clones
            • Stability of training
        """

        plt.figure(figsize=(8,5))
        plt.plot(self.loss_history, label="Loss")
        plt.plot(self.fidB_history, label="Fidelity B")
        plt.plot(self.fidE_history, label="Fidelity E")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Training Dynamics")
        plt.show()