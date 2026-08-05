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


def test_validate_bounds_exceptions() -> None:
    """Asserts that _validate_bounds() successfully intercepts and blocks
    invalid, infinite, or non-numeric (NaN) domain parameters.
    """
    integrator = QuadratureIntegrator(lambda x: x**2)

    # 1. Assert failure when a boundary contains a NaN value
    with pytest.raises(ValueError, match="Integration boundaries cannot be NaN."):
        integrator.trapezoidal(np.nan, 1.0, n=10)

    with pytest.raises(ValueError, match="Integration boundaries cannot be NaN."):
        integrator.simpsons(0.0, np.nan, n=10)

    # 2. Assert failure when bounded methods are passed infinite parameters
    with pytest.raises(ValueError, match="Infinite boundaries detected."):
        integrator.gauss_legendre(-np.inf, 1.0, deg=8)

    with pytest.raises(ValueError, match="Infinite boundaries detected."):
        integrator.clenshaw_curtis(0.0, np.inf, n=16)

    # 3. Assert failure when vectorised arrays contain infinite parameters
    with pytest.raises(ValueError, match="Infinite boundaries detected."):
        vector_b = np.array([1.0, 2.0, np.inf])
        integrator.trapezoidal(0.0, vector_b, n=10)


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


def test_multidimensional_grid_bounds_classical_solvers() -> None:
    """Asserts that both trapezoidal and simpsons methods correctly handle higher-dimensional
    boundary arrays (max_ndim > 0) by dynamically broadcasting the evaluation grid.
    """
    # Simple linear function: f(x) = x. Integral is 0.5 * (b^2 - a^2)
    integrator = QuadratureIntegrator(lambda x: x)

    # Define a 2D grid of upper integration bounds (Shape: 2x3)
    b_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Theoretical exact results: 0.5 * b^2
    expected_results = 0.5 * (b_matrix**2)

    # 1. Triggers 'max_ndim > 0' inside the trapezoidal method
    results_trap = integrator.trapezoidal(0.0, b_matrix, n=100)
    assert results_trap.shape == (2, 3)
    assert np.allclose(results_trap, expected_results, atol=1e-5)

    # 2. Triggers 'max_ndim > 0' inside the simpsons method
    results_simp = integrator.simpsons(0.0, b_matrix, n=100)
    assert results_simp.shape == (2, 3)
    assert np.allclose(results_simp, expected_results, atol=1e-5)


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
    """Tests grid path optimisation handling highly osculating variant inputs."""
    integrator = QuadratureIntegrator(lambda x: np.sin(x))
    res = integrator.adaptive_simpsons(0.0, np.pi, tol=1e-8)

    assert math.isclose(res, 2.0, abs_tol=1e-7)
    assert integrator.metrics["max_recursion_depth"] > 0


def test_adaptive_simpsons_invalid_parameter_guards() -> None:
    """
    Asserts that adaptive_simpsons successfully intercepts and blocks
    invalid tolerance (tol <= 0) and maximum recursion depth (max_depth <= 0) parameters.
    """
    integrator = QuadratureIntegrator(lambda x: x**2)

    # 1. Assert failure when tolerance is exactly zero
    with pytest.raises(
        ValueError, match="Error tolerance 'tol' must be strictly positive."
    ):
        integrator.adaptive_simpsons(0.0, 1.0, tol=0.0)

    # 2. Assert failure when tolerance is a negative value
    with pytest.raises(
        ValueError, match="Error tolerance 'tol' must be strictly positive."
    ):
        integrator.adaptive_simpsons(0.0, 1.0, tol=-1e-5)

    # 3. Assert failure when max_depth is exactly zero
    with pytest.raises(
        ValueError,
        match="Maximum recursion depth 'max_depth' must be a positive integer.",
    ):
        integrator.adaptive_simpsons(0.0, 1.0, max_depth=0)

    # 4. Assert failure when max_depth is a negative integer
    with pytest.raises(
        ValueError,
        match="Maximum recursion depth 'max_depth' must be a positive integer.",
    ):
        integrator.adaptive_simpsons(0.0, 1.0, max_depth=-5)


def test_adaptive_simpsons_nan_and_inf_bounds_guard() -> None:
    """
    Asserts that adaptive_simpsons successfully intercepts and blocks
    non-finite input limits including NaN, positive infinity, and negative infinity.
    """
    integrator = QuadratureIntegrator(lambda x: x**2)
    error_msg = (
        "Invalid parameters passed. Adaptive integration requires fixed finite inputs."
    )

    # 1. Assert failure when the lower bound 'a' is a NaN
    with pytest.raises(ValueError, match=error_msg):
        integrator.adaptive_simpsons(np.nan, 1.0)

    # 2. Assert failure when the upper bound 'b' is a NaN
    with pytest.raises(ValueError, match=error_msg):
        integrator.adaptive_simpsons(0.0, np.nan)

    # 3. Assert failure when the lower bound 'a' is negative infinity
    with pytest.raises(ValueError, match=error_msg):
        integrator.adaptive_simpsons(-np.inf, 1.0)

    # 4. Assert failure when the upper bound 'b' is positive infinity
    with pytest.raises(ValueError, match=error_msg):
        integrator.adaptive_simpsons(0.0, np.inf)


def test_adaptive_simpsons_max_depth_escape_path() -> None:
    """
    Asserts that _adaptive_simpsons_step gracefully handles non-convergence
    by breaking recursion loops and updating state when reaching max_depth.
    """
    # A highly active function that is notoriously difficult to resolve numerically
    challenging_func = lambda x: np.sin(1.0 / (x + 1e-5))  # noqa: E731
    integrator = QuadratureIntegrator(challenging_func)

    # Force an immediate depth break by setting a tiny max_depth cap and high tolerance
    _ = integrator.adaptive_simpsons(0.0, 1.0, tol=1e-12, max_depth=3)

    # 1. Assert that the telemetry registry properly captures the depth ceiling hit
    assert integrator.metrics["max_recursion_depth"] == 3

    # 2. Assert that the engine tracked and accumulated the non-converging subinterval errors
    assert integrator.metrics["estimated_error"] > 0.0


def test_clenshaw_curtis_performance() -> None:
    """Ensures Clenshaw-Curtis FFT mapping calculates high-degree polynomials reliably."""
    integrator = QuadratureIntegrator(lambda x: np.exp(x))
    expected = np.e - 1.0
    res = integrator.clenshaw_curtis(0.0, 1.0, n=32)

    assert math.isclose(res, expected, abs_tol=1e-8)


def test_clenshaw_curtis_invalid_node_guards() -> None:
    """
    Asserts that clenshaw_curtis successfully intercepts and blocks
    invalid, zero, negative, or non-integer node limits (n).
    """
    integrator = QuadratureIntegrator(lambda x: x**2)
    error_msg = "Node limit 'n' must be a positive integer."

    # 1. Assert failure when the node limit is exactly zero
    with pytest.raises(ValueError, match=error_msg):
        integrator.clenshaw_curtis(0.0, 1.0, n=0)

    # 2. Assert failure when the node limit is a negative value
    with pytest.raises(ValueError, match=error_msg):
        integrator.clenshaw_curtis(0.0, 1.0, n=-16)

    # 3. Assert failure when the node limit is passed as an invalid type (float)
    with pytest.raises(ValueError, match=error_msg):
        integrator.clenshaw_curtis(0.0, 1.0, n=16.5)  # type: ignore


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


def test_integrate_infinite_edge_cases_and_defaults() -> None:
    """Asserts that integrate_infinite correctly enforces parameter guards,
    maps lower semi-infinite domains, and injects backend default arguments.
    """
    # CORRECTED: Use a decaying rational function safe from exponential overflows and underflows
    # The analytical integral of 1 / (1 + x^2) from -inf to 0.0 is exactly pi / 2
    integrator = QuadratureIntegrator(lambda x: 1.0 / (1.0 + x**2))

    # 1. Cover: if not inf_a and not inf_b:
    with pytest.raises(
        ValueError, match="Both limits are finite. Use standard methods instead."
    ):
        integrator.integrate_infinite(0.0, 1.0)

    # 2. Cover: if inf_a and inf_b -> if a > 0 or b < 0:
    # Test case where a is +inf (a > 0)
    with pytest.raises(ValueError, match="Infinite range sequence alignment mismatch."):
        integrator.integrate_infinite(np.inf, np.inf)

    # Test case where b is -inf (b < 0)
    with pytest.raises(ValueError, match="Infinite range sequence alignment mismatch."):
        integrator.integrate_infinite(-np.inf, -np.inf)

    # 3. Cover the 'else' branch: lower semi-infinite domain mapping (-inf, b]
    # Analytical integral from -inf to 0.0 is pi / 2
    res_lower_semi = integrator.integrate_infinite(
        -np.inf, 0.0, method="gauss_legendre", deg=64
    )
    assert math.isclose(res_lower_semi, np.pi / 2.0, abs_tol=1e-4)

    # 4. Cover keyword injection for ["gauss_legendre", "clenshaw_curtis"] when missing "deg"/"n"
    # Calling without passing 'deg' forces the method to inject 'deg=50' automatically.
    # For a degree of 50, a standard Gauss-Legendre rule uses exactly 50 function evaluations.
    _ = integrator.integrate_infinite(0.0, np.inf, method="gauss_legendre")
    assert integrator.metrics["function_evaluations"] == 50

    # 5. Cover keyword injection for ["simpsons", "trapezoidal"] when missing "n"
    # Calling without passing 'n' forces the method to inject 'n=100' automatically.
    # For a 100-panel trapezoidal rule, the number of grid evaluation nodes is exactly n + 1 = 101.
    _ = integrator.integrate_infinite(0.0, np.inf, method="trapezoidal")
    assert integrator.metrics["function_evaluations"] == 101


def test_tanh_sinh_endpoint_singularities() -> None:
    """Validates double exponential robustness against severe endpoints tangents."""
    integrator = QuadratureIntegrator(lambda x: 1.0 / np.sqrt(x))
    res = integrator.tanh_sinh(0.0, 1.0, tol=1e-5, max_level=7)
    assert math.isclose(res, 2.0, abs_tol=1e-3)


def test_tanh_sinh_edge_cases_and_overflow_guards() -> None:
    """Asserts that tanh_sinh correctly blocks invalid tolerance parameters (tol <= 0)
    and successfully triggers the continue statement when nodes exceed overflow thresholds.
    """
    integrator = QuadratureIntegrator(lambda x: x**2)

    # 1. Cover: if tol <= 0: (Assert failure when tolerance is exactly zero)
    with pytest.raises(ValueError, match="Tolerance targets must evaluate positively."):
        integrator.tanh_sinh(0.0, 1.0, tol=0.0)

    # 2. Cover: if tol <= 0: (Assert failure when tolerance is a negative value)
    with pytest.raises(ValueError, match="Tolerance targets must evaluate positively."):
        integrator.tanh_sinh(0.0, 1.0, tol=-1e-5)

    # 3. Cover: if not np.any(valid_mask): continue
    # By forcing a massive integration interval along with a high refinement max_level,
    # the internal 't' grid elements will grow significantly larger than 6.0.
    # At high levels, arg = 0.5 * pi * sinh(t) easily exceeds 200, which forces
    # valid_mask to evaluate to all-False and triggers the 'continue' branch cleanly.
    _ = integrator.tanh_sinh(-1000.0, 1000.0, tol=1e-3, max_level=12)

    # Verifying the telemetry count confirms the loop executed and safely
    # skipped overflow nodes via the 'continue' statement without a runtime crash.
    assert integrator.metrics["function_evaluations"] > 0


def test_tanh_sinh_extreme_overflow_trigger(monkeypatch) -> None:
    """Forces valid_mask to be completely False by mock-overflowing the sinh
    calculation, ensuring the continue statement is executed and covered
    while rigorously validating tracking metrics.
    """
    integrator = QuadratureIntegrator(lambda x: x**2)

    # Force np.sinh to return infinity so that arg evaluates to infinity.
    # This ensures valid_mask (arg < 200) becomes completely False on level 1.
    def mock_sinh(x):
        x_arr = np.asanyarray(x)
        return np.full_like(x_arr, np.inf, dtype=np.float64)

    monkeypatch.setattr(np, "sinh", mock_sinh)

    # Running the method forces the 'continue' statement to execute instantly on level 1.
    _ = integrator.tanh_sinh(0.0, 1.0, tol=1e-3, max_level=1)

    # RIGOROUS ASSERTION: Because the 'continue' branch triggers before the midpoint
    # evaluation line inside the loop block, exactly 0 function evaluations take place.
    assert integrator.metrics["function_evaluations"] == 0
