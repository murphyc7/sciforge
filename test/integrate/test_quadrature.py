"""Tests for murphyc7/sciforge/src/sciforge/integrate/quadrature.py"""

import math

import numpy as np
import pytest

from sciforge.integrate.quadrature import QuadratureIntegrator

#######################
# 1. Exception guards #
#######################


def test_initialisation_guard() -> None:
    """Ensures class configuration fails gracefully on incorrect instantiations."""
    with pytest.raises(TypeError):
        QuadratureIntegrator("not_a_callable")  # type: ignore


def test_parameter_validation_guards() -> None:
    """Asserts defensive engineering exception logic triggers across all core workflows."""
    integrator = QuadratureIntegrator(lambda x: np.sin(x))

    with pytest.raises(ValueError, match="even integer"):
        integrator.simpsons(0.0, 1.0, n=5)

    with pytest.raises(ValueError, match="positive integer"):
        integrator.trapezoidal(0.0, 1.0, n=-10)

    with pytest.raises(ValueError, match="positive integer"):
        integrator.gauss_legendre(0.0, 1.0, deg=0)


def test_singularity_and_nan_detection() -> None:
    """Asserts that mathematical poles or exceptions throw safe, controlled processing interrupts."""
    integrator = QuadratureIntegrator(lambda x: 1.0 / x)
    with pytest.raises(ValueError, match="singularity detected"):
        integrator.trapezoidal(0.0, 2.0, n=10)


######################################################################
# 2. Core Classical Solvers (Trapezoidal, Simpson's, Gauss-Legendre) #
######################################################################


def test_trapezoidal_and_simpsons_basic() -> None:
    """Validates baseline convergence accuracy on basic polynomial functions."""
    integrator = QuadratureIntegrator(lambda x: x**2)
    res_trap = integrator.trapezoidal(0.0, 3.0, n=500)
    res_simp = integrator.simpsons(0.0, 3.0, n=100)

    assert math.isclose(res_trap, 9.0, abs_tol=1e-3)
    assert math.isclose(res_simp, 9.0, abs_tol=1e-6)


def test_gauss_legendre_exactness() -> None:
    """Ensures Gauss-Legendre integrates a degree (2N-1) polynomial exactly."""
    integrator = QuadratureIntegrator(lambda x: x**5)
    expected = 64.0 / 6.0
    res = integrator.gauss_legendre(0.0, 2.0, deg=3)
    assert math.isclose(res, expected, abs_tol=1e-12)


###############################################
# 3. Vectorised Bounds and Array Broadcasting #
###############################################


def test_vectorised_bounds_execution() -> None:
    """Verifies internal NumPy dimension broadcasting over concurrent multiple integration domains."""
    integrator = QuadratureIntegrator(lambda x: x**3)
    b_bounds = np.array([1.0, 2.0, 3.0])
    expected = np.array([0.25, 4.0, 20.25])

    res_gl = integrator.gauss_legendre(0.0, b_bounds, deg=8)
    res_cc = integrator.clenshaw_curtis(0.0, b_bounds, n=16)

    assert np.allclose(res_gl, expected, atol=1e-6)
    assert np.allclose(res_cc, expected, atol=1e-6)


###############################################
# 4. Adaptive Refinement and Advanced Solvers #
###############################################


def test_adaptive_simpsons_convergence() -> None:
    """Tests grid path optimization handling highly osculating variant inputs."""
    integrator = QuadratureIntegrator(lambda x: np.sin(x))
    res = integrator.adaptive_simpsons(0.0, np.pi, tol=1e-8)

    assert math.isclose(res, 2.0, abs_tol=1e-7)
    assert integrator.metrics["max_recursion_depth"] > 0


def test_clenshaw_curtis_performance() -> None:
    """Ensures Clenshaw-Curtis FFT mapping calculates high-degree polynomials reliably."""
    integrator = QuadratureIntegrator(lambda x: np.exp(x))
    expected = np.e - 1.0
    res = integrator.clenshaw_curtis(0.0, 1.0, n=32)

    assert math.isclose(res, expected, abs_tol=1e-8)


#####################################################
# 5. Infinite Boundaries and Endpoint Singularities #
#####################################################


def test_infinite_boundary_transformations() -> None:
    """Asserts mapping transforms resolve equations integrating across infinite limits."""
    integrator_gaussian = QuadratureIntegrator(lambda x: np.exp(-(x**2)))
    res_inf = integrator_gaussian.integrate_infinite(
        -np.inf, np.inf, method="gauss_legendre", deg=64
    )
    assert math.isclose(res_inf, np.sqrt(np.pi), abs_tol=1e-4)

    integrator_rational = QuadratureIntegrator(lambda x: 1.0 / (1.0 + x**2))
    res_semi = integrator_rational.integrate_infinite(
        0.0, np.inf, method="clenshaw_curtis", n=64
    )
    assert math.isclose(res_semi, np.pi / 2.0, abs_tol=1e-4)


def test_tanh_sinh_endpoint_singularities() -> None:
    """Validates double exponential robustness against severe endpoints tangents."""
    integrator = QuadratureIntegrator(lambda x: 1.0 / np.sqrt(x))
    res = integrator.tanh_sinh(0.0, 1.0, tol=1e-5, max_level=7)
    assert math.isclose(res, 2.0, abs_tol=1e-3)
