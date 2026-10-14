r"""Classical reference solutions for validating Physics-Informed Neural Network output.

Every solver here is constructed from the *same* physics objects the PINN trains against,
so material constants and governing equations cannot silently drift apart. The reference
profiles are expressed in the PINN's normalised coordinates, where :math:`x \in [0, 1]`.

Two references are provided:

- **Constant field:** the exact analytic solution of the 1D drift-diffusion equation.
- **Coupled Poisson:** a numerical solution of the self-consistent Poisson and
  drift-diffusion system via :func:`scipy.integrate.solve_bvp`, which controls its own
  error tolerance through adaptive mesh refinement.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)


@dataclass(frozen=True)
class ReferenceSolution:
    r"""A classical reference profile evaluated on a spatial grid.

    Attributes:
        x (np.ndarray): Normalised spatial coordinates in [0, 1], shape [N].
        n (np.ndarray): Normalised carrier density :math:`n(x)`, shape [N].
        dn_dx (np.ndarray): Spatial derivative of ``n`` in normalised units, shape [N].
        phi (np.ndarray | None): Electrostatic potential :math:`\phi(x)` in volts, shape
            [N]. None for solvers that do not resolve the potential. Defaults to None.
        dphi_dx (np.ndarray | None): Spatial derivative of ``phi``, shape [N]. None when
            ``phi`` is None. Defaults to None.
    """

    x: np.ndarray
    n: np.ndarray
    dn_dx: np.ndarray
    phi: np.ndarray | None = None
    dphi_dx: np.ndarray | None = None


def _resolve_grid(x: np.ndarray | None, default_points: int = 201) -> np.ndarray:
    """Validates a user-supplied evaluation grid, or builds a uniform default.

    Args:
        x (np.ndarray | None): Requested evaluation coordinates, or None for a uniform grid.
        default_points (int, optional): Size of the default grid. Defaults to 201.

    Returns:
        np.ndarray: A one-dimensional float64 grid contained in [0, 1].

    Raises:
        ValueError: If the grid is empty, is not one-dimensional, or leaves [0, 1].
    """
    if x is None:
        return np.linspace(0.0, 1.0, default_points)

    grid = np.asarray(x, dtype=np.float64).ravel()
    if grid.size == 0:
        raise ValueError("Evaluation grid must contain at least one point.")
    if grid.min() < 0.0 or grid.max() > 1.0:
        raise ValueError(
            "Evaluation grid must lie inside the normalised domain [0, 1]."
        )
    return grid


def solve_constant_field(
    physics: ConstantFieldPhysics,
    n_left: float = 1.0,
    n_right: float = 0.0,
    x: np.ndarray | None = None,
) -> ReferenceSolution:
    r"""Evaluates the exact solution of the constant-field drift-diffusion equation.

    In normalised coordinates the transport residual reduces to
    :math:`n'' + \mathrm{Pe}\, n' = 0` with the Péclet number
    :math:`\mathrm{Pe} = E L q / (k_B T)`, whose solution under Dirichlet boundaries is

    .. math::

       n(x) = n_R + (n_L - n_R)\,\frac{e^{-\mathrm{Pe}\,x} - e^{-\mathrm{Pe}}}
       {1 - e^{-\mathrm{Pe}}}

    Args:
        physics (ConstantFieldPhysics): The solver whose constants define the problem.
        n_left (float, optional): Boundary density at x = 0. Defaults to 1.0.
        n_right (float, optional): Boundary density at x = 1. Defaults to 0.0.
        x (np.ndarray | None, optional): Evaluation grid in [0, 1]. Defaults to a uniform
            201-point grid.

    Returns:
        ReferenceSolution: The density profile and its derivative (no potential).
    """
    grid = _resolve_grid(x)
    peclet = physics.electric_field * physics.L_scale / (physics.kb_t / physics.q)
    span = n_left - n_right

    if abs(peclet) < 1.0e-12:
        # Pure diffusion limit: the exponential degenerates to a straight line.
        n = n_right + span * (1.0 - grid)
        dn_dx = np.full_like(grid, -span)
    else:
        # expm1 keeps the ratio accurate when the Péclet number is small.
        denominator = -np.expm1(-peclet)
        n = (
            n_right
            + span * (np.expm1(-peclet * grid) - np.expm1(-peclet)) / denominator
        )
        dn_dx = -span * peclet * np.exp(-peclet * grid) / denominator

    return ReferenceSolution(x=grid, n=n, dn_dx=dn_dx)


def solve_coupled_poisson(
    physics: PoissonCoupledPhysics,
    n_left: float = 1.0,
    n_right: float = 0.1,
    phi_left: float = 0.0,
    phi_right: float = 0.5,
    x: np.ndarray | None = None,
    tol: float = 1.0e-8,
    max_nodes: int = 200_000,
) -> ReferenceSolution:
    r"""Solves the self-consistent Poisson and drift-diffusion boundary value problem.

    Solves exactly the system encoded by ``PoissonCoupledPhysics.compute_residuals``:

    .. math::

       \phi'' = -K\,(N_D - n), \qquad
       \beta\, n'' - \phi'\, n' - \phi''\, n = 0

    with :math:`K = L^2 q N / (\epsilon_0 \epsilon_r)` and :math:`\beta = k_B T / q`. The
    transport equation is a statement of current continuity,
    :math:`(\beta n' - \phi' n)' = 0`, so the current is constant across the domain.

    Args:
        physics (PoissonCoupledPhysics): The solver whose constants define the problem.
        n_left (float, optional): Boundary density at x = 0. Defaults to 1.0.
        n_right (float, optional): Boundary density at x = 1. Defaults to 0.1.
        phi_left (float, optional): Boundary potential at x = 0. Defaults to 0.0.
        phi_right (float, optional): Boundary potential at x = 1. Defaults to 0.5.
        x (np.ndarray | None, optional): Evaluation grid in [0, 1]. Defaults to a uniform
            201-point grid.
        tol (float, optional): Relative residual tolerance for the BVP solver.
            Defaults to 1e-8.
        max_nodes (int, optional): Maximum adaptive mesh size. Defaults to 200,000.

    Returns:
        ReferenceSolution: Density and potential profiles with their derivatives.

    Raises:
        ValueError: If the evaluation grid is invalid.
        RuntimeError: If the BVP solver does not converge to the requested tolerance.
    """
    grid = _resolve_grid(x)

    # These expressions mirror PoissonCoupledPhysics.compute_residuals. The consistency
    # test that feeds this reference back through that method guards against drift.
    poisson_constant = (
        (physics.L_scale**2)
        * (physics.q * physics.N_scale)
        / (physics.eps0 * physics.eps_r)
    )
    beta = physics.kb_t / physics.q
    nd_scaled = physics.nd_scaled

    def rhs(_x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """State vector y = [n, n', phi, phi']; returns its derivative."""
        n, dn_dx, _phi, dphi_dx = y
        d2phi_dx2 = -poisson_constant * (nd_scaled - n)
        d2n_dx2 = (dphi_dx * dn_dx + d2phi_dx2 * n) / beta
        return np.vstack([dn_dx, d2n_dx2, dphi_dx, d2phi_dx2])

    def boundary_residual(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array(
            [ya[0] - n_left, yb[0] - n_right, ya[2] - phi_left, yb[2] - phi_right]
        )

    # Linear initial guess that satisfies the boundary values
    mesh = np.linspace(0.0, 1.0, 201)
    initial_guess = np.vstack(
        [
            n_left + (n_right - n_left) * mesh,
            np.full_like(mesh, n_right - n_left),
            phi_left + (phi_right - phi_left) * mesh,
            np.full_like(mesh, phi_right - phi_left),
        ]
    )

    solution = solve_bvp(
        rhs, boundary_residual, mesh, initial_guess, tol=tol, max_nodes=max_nodes
    )
    if not solution.success:
        raise RuntimeError(f"solve_bvp did not converge: {solution.message}")

    n, dn_dx, phi, dphi_dx = solution.sol(grid)
    return ReferenceSolution(x=grid, n=n, dn_dx=dn_dx, phi=phi, dphi_dx=dphi_dx)
