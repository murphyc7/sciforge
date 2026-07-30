"""Multi-Layer Perceptron architectures optimised for differential calculus solvers.

Defines smooth multilayer network profiles designed to support uninterrupted backpropagation
operations without suffering from vanishing numerical gradients.
"""

import torch
import torch.nn as nn


class CarrierPINN(nn.Module):
    """A deep network layer architecture for modeling charge carrier spatial distributions.

    Implements a fully connected Multi-Layer Perceptron (MLP) configuration using Hyperbolic
    Tangent (Tanh) activation functions to calculate stable higher-order derivatives.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_layers: int = 4,
        hidden_dim: int = 50,
    ) -> None:
        """Initializes the network layer architecture.

        Args:
            input_dim (int, optional): Coordinate parameter input counts. Defaults to 1.
            output_dim (int, optional): Concentration parameter targets. Defaults to 1.
            hidden_layers (int, optional): Total processing layer depths. Defaults to 4.
            hidden_dim (int, optional): Processing channel counts per layer. Defaults to 50.
        """
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
        """Evaluates carrier concentration predictions at targeted spatial grids.

        Args:
            x (torch.Tensor): Coordinates tensor grid of shape [Batch Size, 1].

        Returns:
            torch.Tensor: The corresponding predicted carrier concentrations.
        """
        return self.network(x)
