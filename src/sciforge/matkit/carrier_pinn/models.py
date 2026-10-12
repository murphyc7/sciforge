"""Multi-Layer Perceptron architectures optimised for differential calculus solvers.

Defines smooth multilayer network profiles designed to support uninterrupted backpropagation
operations without suffering from vanishing numerical gradients.
"""

import torch


class CarrierPINN(torch.nn.Module):
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
        """Initialises the network layer architecture.

        Args:
            input_dim (int, optional): Coordinate parameter input counts. Defaults to 1.
            output_dim (int, optional): Configurable nodes. Use 1 for constant, 2 for coupled,
                and 3 for bipolar solvers. Defaults to 1.
            hidden_layers (int, optional): Total processing layer depths. Defaults to 4.
            hidden_dim (int, optional): Processing channels per layer. Defaults to 50.
        """
        super().__init__()

        layers: list[torch.nn.Module] = []
        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.Tanh())

        # Output layer (Predicting carrier concentration 'n')
        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates carrier concentration predictions at targeted spatial grids.

        Args:
            x (torch.Tensor): Coordinates tensor grid of shape [Batch Size, 1].

        Returns:
            predictions (torch.Tensor): The corresponding predicted carrier concentrations.
        """
        predictions: torch.Tensor = self.network(x)
        return predictions
