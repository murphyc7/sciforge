import torch
import torch.nn as nn


class CarrierPINN(nn.Module):
    """Multi-layer Perceptron optimised for Physics-Informed Neural Network evaluation."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_layers: int = 4,
        hidden_dim: int = 50,
    ) -> None:
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer (Predicting carrier concentration 'n')
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates carrier concentration n(x) at spatial position x."""
        return self.network(x)
