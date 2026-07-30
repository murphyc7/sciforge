"""Tests for murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/"""

import torch

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import DriftDiffusionPhysics

###################################################################
# 1. murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/models.py #
###################################################################


def test_pinn_model_initialization_and_forward_shapes() -> None:
    """Verifies that network dimensions, tensor transforms, and activation shapes map properly."""
    # Arrange: Initialize a standard default model layout
    model = CarrierPINN(input_dim=1, output_dim=1, hidden_layers=3, hidden_dim=20)

    # Create a mock batch tensor representing spatial coordinates: Shape [Batch, SpatialDimension]
    test_input = torch.linspace(0.0, 1.0, 15, dtype=torch.float32).view(-1, 1)

    # Act: Compute the forward transformation pass
    test_output = model(test_input)

    # Assert: Confirm vector dimensions match the expected target dimensions [15, 1]
    assert test_output.shape == (15, 1)
    assert test_output.dtype == torch.float32


####################################################################
# 2. murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/physics.py #
####################################################################


def test_physics_engine_mobility_scaling() -> None:
    """Confirms that material mobility scales inversely with effective mass configurations."""
    # Case A: Lightweight effective mass profile (e.g., GaAs-like)
    light_physics = DriftDiffusionPhysics(effective_mass=0.067, permittivity=12.9)

    # Case B: Heavy effective mass profile (e.g., a heavy-hole tier)
    heavy_physics = DriftDiffusionPhysics(effective_mass=0.5, permittivity=11.5)

    # Assert: Mobility must follow explicit physical inverse proportionality guidelines
    assert light_physics.mobility > heavy_physics.mobility
    assert light_physics.diffusion_coeff > heavy_physics.diffusion_coeff


def test_pde_residual_gradient_computation_and_autograd_graph() -> None:
    """Guarantees that autograd successfully hooks partial differential gradients to the loss graph."""
    # Arrange: Setup an isolated network layer and physics configuration
    model = CarrierPINN(input_dim=1, output_dim=1, hidden_dim=10)
    physics_engine = DriftDiffusionPhysics(effective_mass=0.1, permittivity=10.0)

    # Create coordinate grids that explicitly require backpropagation graph histories
    x_grid = torch.linspace(0.1, 0.9, 10, dtype=torch.float32).view(-1, 1)

    # Act: Evaluate spatial partial differential graph residuals
    residual = physics_engine.compute_pde_residual(x_grid, model)

    # Assert: Verify outputs are structural vectors maintaining graph gradients
    assert residual.shape == (10, 1)
    assert (
        residual.requires_grad is True
    )  # Crucial! Verifies that .backward() can track this tensor
