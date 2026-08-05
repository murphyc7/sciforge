"""Tests for murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/"""

import torch

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)

###################################################################
# 1. murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/models.py #
###################################################################


def test_carrier_pinn_scales_output_channels_dynamically() -> None:
    """Verifies that the network layer scales its output shape based on output_dim configurations."""
    # Arrange: Initialise models mirroring both constant and coupled simulation requirements
    model_constant = CarrierPINN(input_dim=1, output_dim=1, hidden_dim=16)
    model_coupled = CarrierPINN(input_dim=1, output_dim=2, hidden_dim=16)

    # Generate a dummy input matrix slice representing an evaluation point
    test_coordinate = torch.tensor([[0.5]], dtype=torch.float32)

    # Act: Run forward propagation loops
    output_constant = model_constant(test_coordinate)
    output_coupled = model_coupled(test_coordinate)

    # Assert: Verify boundary shapes and precision layers match exact expected metrics
    assert output_constant.shape == (1, 1)
    assert output_coupled.shape == (1, 2)
    assert output_constant.dtype == torch.float32
    assert output_coupled.dtype == torch.float32


####################################################################
# 2. murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/physics.py #
####################################################################


def test_constant_field_physics_evaluates_valid_transport_gradients() -> None:
    """Confirms ConstantFieldPhysics evaluates 1D transport residuals with a valid gradient history."""
    # Arrange: Build an architecture matching the single-field constant framework layout
    model = CarrierPINN(input_dim=1, output_dim=1, hidden_dim=10)
    solver = ConstantFieldPhysics(
        effective_mass=0.067, permittivity=12.9, electric_field=1.0e3
    )

    # Generate an arbitrary coordinate grid evaluation tensor matrix
    x_grid = torch.linspace(0.0, 1.0, 12, dtype=torch.float32).view(-1, 1)

    # Act: Compute the physics residuals using the strategy pattern interface
    residuals = solver.compute_residuals(x_grid, model)

    # Assert: Check dictionary keys, data structures, and mathematical backprop tracking
    assert "transport" in residuals
    assert (
        "poisson" not in residuals
    )  # Constant engine must not trigger Poisson equations
    assert residuals["transport"].shape == (12, 1)
    assert (
        residuals["transport"].requires_grad is True
    )  # Enforces backpropagation capability


def test_poisson_coupled_physics_evaluates_multi_field_residuals() -> None:
    """Confirms PoissonCoupledPhysics evaluates both Poisson and transport continuity matrices."""
    # Arrange: Build an architecture matching the multi-field coupled framework layout
    model = CarrierPINN(input_dim=1, output_dim=2, hidden_dim=12)
    solver = PoissonCoupledPhysics(
        effective_mass=0.1, permittivity=11.9, donor_doping=1.0e22
    )

    # Generate a tight coordinate space tracking grid
    x_grid = torch.linspace(0.0, 1.0e-6, 15, dtype=torch.float32).view(-1, 1)

    # Act: Compute multi-field system residuals simultaneously
    residuals = solver.compute_residuals(x_grid, model)

    # Assert: Confirm that both physical equations evaluate distinct matrix blocks successfully
    assert "poisson" in residuals
    assert "transport" in residuals

    assert residuals["poisson"].shape == (15, 1)
    assert residuals["transport"].shape == (15, 1)

    assert residuals["poisson"].requires_grad is True
    assert residuals["transport"].requires_grad is True
