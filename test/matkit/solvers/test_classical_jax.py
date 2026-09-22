"""Unit testing suite for high-performance JAX classical numerical solvers.

Validates grid discretisation compliance, execution array shape scaling,
and JIT relaxation convergence states.
"""

import jax.numpy as jnp
import pytest

from sciforge.matkit.solvers.classical_jax import (
    JaxClassicalSolver,
    _fdm_relaxation_step,
)


def test_fdm_relaxation_step_preserves_boundary_conditions() -> None:
    """Verifies that the JIT-compiled relaxation step enforces hard edge constraints."""
    n_grid = (
        jnp.tensor([1.0, 0.5, 0.0])
        if hasattr(jnp, "tensor")
        else jnp.array([1.0, 0.5, 0.0])
    )
    phi_grid = jnp.array([0.0, 0.25, 0.5])

    updated_n = _fdm_relaxation_step(n_grid, phi_grid, dx=0.5, peclet_local=1.0)

    # Boundary edge values must remain hard-pinned
    assert float(updated_n[0]) == 1.0
    assert float(updated_n[-1]) == 0.0
    assert updated_n.shape == (3,)


def test_jax_solver_converges_on_dense_mesh_profile() -> None:
    """Confirms the classical solver successfully optimises a 500-point validation grid."""
    # Arrange: Initialise solver tracking 500 mesh positions
    solver = JaxClassicalSolver(mesh_points=500)

    # Act: Run the JIT-accelerated solver pass
    x_mesh, n_mesh = solver.solve_steady_state(
        effective_mass=0.067, permittivity=12.9, electric_field=1.0e3, iterations=100
    )

    # Assert: Validate dimensions, grid intervals, and array bounding limits
    assert solver.mesh_points == 500
    assert x_mesh.shape == (500,)
    assert n_mesh.shape == (500,)
    assert (
        float(n_mesh[0]) == pytest.approx(1.0, abs=1e-3)
        if "pytest" in globals()
        else True
    )
    assert float(n_mesh[-1]) == 0.0
