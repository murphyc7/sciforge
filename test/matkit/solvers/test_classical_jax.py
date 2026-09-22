"""Tests for murphyc7/sciforge/src/sciforge/matkit/solvers/classical_jax.py"""

import jax.numpy as jnp
import pytest

from sciforge.matkit.solvers.classical_jax import (
    JaxClassicalSolver,
    _fdm_transport_relaxation_step,
    _solve_poisson_matrix,
)


def test_solve_poisson_matrix_dimensions_and_boundaries() -> None:
    """Verifies that the JIT Poisson matrix inversion maps correct shapes and boundary flags."""
    n_grid = jnp.linspace(1.0, 0.1, 50)
    phi = _solve_poisson_matrix(n_grid, dx=0.02, gamma=10.0, nd=1.0)

    assert phi.shape == (50,)
    assert float(phi[0]) == 0.0
    assert float(phi[-1]) == 0.5


def test_fdm_transport_relaxation_step_boundaries() -> None:
    """Verifies that the central-difference transport relaxation step preserves boundary limits."""
    n_grid = jnp.linspace(1.0, 0.1, 50)
    phi_grid = jnp.linspace(0.0, 0.5, 50)

    n_new = _fdm_transport_relaxation_step(n_grid, phi_grid, dx=0.02, beta=0.0259)

    assert n_new.shape == (50,)
    assert float(n_new[0]) == 1.0
    assert float(n_new[-1]) == 0.1


def test_jax_solver_factory_processes_all_engine_modes() -> None:
    """Confirms the classical solver successfully optimises 500-point multi-engine validation grids."""
    solver = JaxClassicalSolver(mesh_points=500)

    # 1. Test Constant Field Engine Path
    x_c, n_c, phi_c = solver.solve_steady_state(
        engine_mode="constant", effective_mass=0.067, permittivity=12.9, iterations=5
    )
    assert x_c.shape == (500,)
    assert n_c.shape == (500,)
    assert float(phi_c[0]) == 0.0

    # 2. Test Coupled Poisson Engine Path
    x_p, n_p, phi_p = solver.solve_steady_state(
        engine_mode="coupled", effective_mass=0.067, permittivity=12.9, iterations=5
    )
    assert x_p.shape == (500,)
    assert n_p.shape == (500,)
    assert phi_p.shape == (500,)
    assert float(n_p[0]) == pytest.approx(1.0, abs=1e-3)
    assert float(n_p[-1]) == pytest.approx(0.1, abs=1e-3)
