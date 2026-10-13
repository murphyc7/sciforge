"""Analytic (manufactured-solution) tests for murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/physics.py residuals.

Each test substitutes a deterministic, closed-form network for the trained PINN, so the
expected residual can be derived by hand and compared against the autograd result.
"""

import pytest
import torch

from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)

##############################################
# 0. Closed-form stand-ins for a trained net #
##############################################


class _ExponentialField(torch.nn.Module):
    """n(x) = A * exp(-k * x) + C, an exact solution of n'' + k * n' = 0."""

    def __init__(self, decay: float) -> None:
        super().__init__()
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.exp(-self.decay * x) + 0.5


class _QuadraticField(torch.nn.Module):
    """n(x) = x^2, so n' = 2x and n'' = 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class _CoupledManufacturedField(torch.nn.Module):
    """Returns [n, phi] with n = N_D + x and phi = K x^3 / 6 (so phi'' = K x)."""

    def __init__(self, nd_scaled: float, poisson_constant: float) -> None:
        super().__init__()
        self.nd_scaled = nd_scaled
        self.poisson_constant = poisson_constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.nd_scaled + x
        phi = self.poisson_constant * x**3 / 6.0
        return torch.cat([n, phi], dim=1)


def _grid() -> torch.Tensor:
    """Float64 evaluation grid so autograd round-off stays far below test tolerances."""
    return torch.linspace(0.0, 1.0, 21, dtype=torch.float64).view(-1, 1)


def _peclet(solver: ConstantFieldPhysics) -> float:
    """Dimensionless drift strength q * E * L / (kB * T) implied by the solver constants."""
    return solver.electric_field * solver.L_scale / (solver.kb_t / solver.q)


##############################################
# 1. ConstantFieldPhysics: n'' + Pe * n' = 0 #
##############################################


def test_constant_field_residual_vanishes_for_exact_exponential_solution() -> None:
    """The normalised residual is n'' + Pe n', so n = A exp(-Pe x) + C must give zero."""
    # Arrange: A strong field makes the drift term non-trivial
    solver = ConstantFieldPhysics(
        effective_mass=0.067, permittivity=12.9, electric_field=5.0e4
    )
    model = _ExponentialField(decay=_peclet(solver))

    # Act
    residual = solver.compute_residuals(_grid(), model)["transport"]

    # Assert
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-9)


def test_constant_field_residual_matches_hand_derived_value_for_quadratic() -> None:
    """For n = x^2 the normalised residual must equal 2 + 2 * Pe * x exactly."""
    # Arrange
    solver = ConstantFieldPhysics(
        effective_mass=0.067, permittivity=12.9, electric_field=5.0e4
    )
    x = _grid()
    expected = 2.0 + 2.0 * _peclet(solver) * x.detach()

    # Act
    residual = solver.compute_residuals(x, _QuadraticField())["transport"]

    # Assert
    assert torch.allclose(residual, expected, rtol=1e-9, atol=1e-9)


def test_constant_field_normalised_residual_is_independent_of_effective_mass() -> None:
    """Mobility and diffusion scale together, so the normalised residual must not depend on mass."""
    # Arrange
    light = ConstantFieldPhysics(effective_mass=0.067, permittivity=12.9)
    heavy = ConstantFieldPhysics(effective_mass=1.08, permittivity=11.7)

    # Act
    residual_light = light.compute_residuals(_grid(), _QuadraticField())["transport"]
    residual_heavy = heavy.compute_residuals(_grid(), _QuadraticField())["transport"]

    # Assert
    assert torch.allclose(residual_light, residual_heavy, rtol=1e-8, atol=1e-10)


#################################################
# 2. PoissonCoupledPhysics: manufactured fields #
#################################################


def test_poisson_coupled_residuals_match_manufactured_solution() -> None:
    """With n = N_D + x and phi = K x^3 / 6, Poisson is exactly 0 and transport is -mu K (1.5 x^2 + N_D x)."""
    # Arrange
    solver = PoissonCoupledPhysics(effective_mass=0.067, permittivity=12.9)
    poisson_constant = (
        (solver.L_scale**2) * (solver.q * solver.N_scale) / (solver.eps0 * solver.eps_r)
    )
    model = _CoupledManufacturedField(solver.nd_scaled, poisson_constant)
    x = _grid()
    expected_transport = (
        -solver.mobility
        * poisson_constant
        * (1.5 * x.detach() ** 2 + solver.nd_scaled * x.detach())
    )

    # Act
    residuals = solver.compute_residuals(x, model)

    # Assert
    assert set(residuals) == {"poisson", "transport"}
    assert torch.allclose(
        residuals["poisson"], torch.zeros_like(residuals["poisson"]), atol=1e-9
    )
    assert torch.allclose(
        residuals["transport"], expected_transport, rtol=1e-8, atol=1e-9
    )


def test_poisson_residual_responds_linearly_to_charge_imbalance() -> None:
    """Perturbing n away from N_D with a flat potential must give residual K * (N_D - n)."""
    # Arrange: phi = 0 everywhere, n = N_D + 0.25 (a uniform surplus of carriers)
    solver = PoissonCoupledPhysics(effective_mass=0.067, permittivity=12.9)
    poisson_constant = (
        (solver.L_scale**2) * (solver.q * solver.N_scale) / (solver.eps0 * solver.eps_r)
    )

    class _FlatPotential(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # The zero-weighted x**2 terms keep a second-order autograd graph alive
            # without changing the field values (a bare constant has no grad_fn).
            n = solver.nd_scaled + 0.25 + 0.0 * x**2
            return torch.cat([n, 0.0 * x**2], dim=1)

    # Act
    residual = solver.compute_residuals(_grid(), _FlatPotential())["poisson"]

    # Assert
    expected = torch.full_like(residual, -0.25 * poisson_constant)
    assert torch.allclose(residual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("electric_field", [1.0e2, 1.0e4, 1.0e6])
def test_constant_field_residual_scales_linearly_with_field_for_unit_slope(
    electric_field: float,
) -> None:
    """For n = x the residual reduces to Pe, which must grow linearly with the applied field."""
    # Arrange
    solver = ConstantFieldPhysics(
        effective_mass=0.067, permittivity=12.9, electric_field=electric_field
    )

    class _LinearField(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # The zero-weighted x**2 term keeps a second-order autograd graph alive
            # without changing the field values (n' = 1 alone has no grad_fn).
            return x + 0.0 * x**2

    # Act
    residual = solver.compute_residuals(_grid(), _LinearField())["transport"]

    # Assert
    expected = torch.full_like(residual, _peclet(solver))
    assert torch.allclose(residual, expected, rtol=1e-8, atol=1e-12)
