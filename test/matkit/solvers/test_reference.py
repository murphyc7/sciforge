"""Tests for murphyc7/sciforge/src/sciforge/matkit/solvers/reference.py"""

import numpy as np
import pytest
import torch

from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
from sciforge.matkit.solvers.reference import (
    ReferenceSolution,
    solve_constant_field,
    solve_coupled_poisson,
)

#########################################################
# 0. Differentiable interpolant of a reference solution #
#########################################################


class _HermiteField(torch.nn.Module):
    """Cubic-Hermite interpolant built from a reference's values and slopes.

    Lets the repository's own autograd residuals be evaluated on a classical solution,
    which is the strongest check that the reference solves the same PDE as the PINN.
    """

    def __init__(self, reference: ReferenceSolution) -> None:
        super().__init__()
        self.x = torch.tensor(reference.x, dtype=torch.float64)
        values = [reference.n]
        slopes = [reference.dn_dx]
        if reference.phi is not None and reference.dphi_dx is not None:
            values.append(reference.phi)
            slopes.append(reference.dphi_dx)
        self.values = [torch.tensor(v, dtype=torch.float64) for v in values]
        self.slopes = [torch.tensor(s, dtype=torch.float64) for s in slopes]

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = query.squeeze(-1)
        i = (torch.searchsorted(self.x, q.detach(), right=True) - 1).clamp(
            0, self.x.numel() - 2
        )
        h = self.x[i + 1] - self.x[i]
        t = (q - self.x[i]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        columns = [
            h00 * v[i] + h10 * h * d[i] + h01 * v[i + 1] + h11 * h * d[i + 1]
            for v, d in zip(self.values, self.slopes, strict=True)
        ]
        return torch.stack(columns, dim=1)


def _query_points() -> torch.Tensor:
    """Interior query points, kept clear of the interpolant's end knots."""
    return torch.linspace(0.01, 0.99, 300, dtype=torch.float64).view(-1, 1)


############################
# 1. Constant-field solver #
############################


def test_constant_field_reference_satisfies_boundary_values() -> None:
    """Verifies the analytic profile hits both Dirichlet boundary values exactly."""
    # Arrange
    solver = ConstantFieldPhysics(0.067, 12.9, electric_field=5.0e4)

    # Act
    reference = solve_constant_field(solver, n_left=1.0, n_right=0.25)

    # Assert
    assert reference.n[0] == pytest.approx(1.0, abs=1e-12)
    assert reference.n[-1] == pytest.approx(0.25, abs=1e-12)
    assert reference.phi is None


def test_constant_field_reference_has_zero_residual_under_repository_physics() -> None:
    """Feeds the analytic solution back through ConstantFieldPhysics; the residual must vanish."""
    # Arrange: A strong field makes the drift term non-trivial
    solver = ConstantFieldPhysics(0.067, 12.9, electric_field=5.0e4)
    field = _HermiteField(solve_constant_field(solver, x=np.linspace(0, 1, 8001)))

    # Act
    residual = solver.compute_residuals(_query_points(), field)["transport"]

    # Assert
    assert residual.abs().max().item() < 1e-5


def test_constant_field_reference_reduces_to_linear_profile_without_field() -> None:
    """Confirms the zero-field diffusion limit is a straight line, not a 0/0 division."""
    # Arrange
    solver = ConstantFieldPhysics(0.067, 12.9, electric_field=0.0)
    grid = np.linspace(0.0, 1.0, 11)

    # Act
    reference = solve_constant_field(solver, n_left=1.0, n_right=0.0, x=grid)

    # Assert
    np.testing.assert_allclose(reference.n, 1.0 - grid, atol=1e-12)
    np.testing.assert_allclose(reference.dn_dx, -1.0, atol=1e-12)


def test_constant_field_reference_is_monotonic_decay_for_positive_field() -> None:
    """Checks the profile falls monotonically from the high boundary to the low boundary."""
    # Arrange
    solver = ConstantFieldPhysics(0.067, 12.9, electric_field=5.0e4)

    # Act
    reference = solve_constant_field(solver)

    # Assert
    assert np.all(np.diff(reference.n) < 0.0)


##############################
# 2. Coupled Poisson solver  #
##############################


def test_coupled_reference_satisfies_all_four_boundary_values() -> None:
    """Verifies density and potential both meet their Dirichlet values at each end."""
    # Arrange
    solver = PoissonCoupledPhysics(0.067, 12.9)

    # Act
    reference = solve_coupled_poisson(solver)

    # Assert
    assert reference.phi is not None
    assert reference.n[0] == pytest.approx(1.0, abs=1e-6)
    assert reference.n[-1] == pytest.approx(0.1, abs=1e-6)
    assert reference.phi[0] == pytest.approx(0.0, abs=1e-6)
    assert reference.phi[-1] == pytest.approx(0.5, abs=1e-6)


def test_coupled_reference_has_zero_residual_under_repository_physics() -> None:
    """Feeds the BVP solution through PoissonCoupledPhysics; both residuals must vanish."""
    # Arrange
    solver = PoissonCoupledPhysics(0.067, 12.9)
    field = _HermiteField(solve_coupled_poisson(solver, x=np.linspace(0, 1, 8001)))

    # Act
    residuals = solver.compute_residuals(_query_points(), field)

    # Assert: Orders of magnitude below the ~12 a wrong field produces
    assert residuals["poisson"].abs().max().item() < 1e-3
    assert residuals["transport"].abs().max().item() < 1e-2


def test_coupled_reference_conserves_current() -> None:
    """Checks the physical invariant: beta*n' - phi'*n is constant across the domain."""
    # Arrange
    solver = PoissonCoupledPhysics(0.067, 12.9)
    beta = solver.kb_t / solver.q

    # Act
    reference = solve_coupled_poisson(solver, x=np.linspace(0, 1, 500))
    assert reference.phi is not None and reference.dphi_dx is not None
    current = beta * reference.dn_dx - reference.dphi_dx * reference.n

    # Assert
    assert np.ptp(current) < 1e-6


def test_coupled_reference_raises_when_solver_cannot_converge() -> None:
    """Ensures a failed BVP solve surfaces as an error instead of returning garbage."""
    # Arrange: A mesh budget far below the 201-node starting mesh cannot converge
    solver = PoissonCoupledPhysics(0.067, 12.9)

    # Act / Assert
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_coupled_poisson(solver, max_nodes=5)


####################
# 3. Grid handling #
####################


@pytest.mark.parametrize(
    "bad_grid", [np.array([-0.1, 0.5]), np.array([0.5, 1.2]), np.array([])]
)
def test_reference_solvers_reject_grids_outside_the_normalised_domain(
    bad_grid: np.ndarray,
) -> None:
    """Verifies empty grids and points outside [0, 1] are rejected by both solvers."""
    # Arrange
    constant = ConstantFieldPhysics(0.067, 12.9)
    coupled = PoissonCoupledPhysics(0.067, 12.9)

    # Act / Assert
    with pytest.raises(ValueError):
        solve_constant_field(constant, x=bad_grid)
    with pytest.raises(ValueError):
        solve_coupled_poisson(coupled, x=bad_grid)


def test_reference_solvers_evaluate_on_the_requested_grid() -> None:
    """Confirms returned arrays match the caller's grid, including non-uniform spacing."""
    # Arrange
    grid = np.array([0.0, 0.1, 0.35, 0.8, 1.0])

    # Act
    constant = solve_constant_field(ConstantFieldPhysics(0.067, 12.9), x=grid)
    coupled = solve_coupled_poisson(PoissonCoupledPhysics(0.067, 12.9), x=grid)

    # Assert
    for reference in (constant, coupled):
        np.testing.assert_array_equal(reference.x, grid)
        assert reference.n.shape == grid.shape
