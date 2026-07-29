"""Tests for murphyc7/sciforge/src/sciforge/integrate/quadrature.py"""

import pytest

from sciforge.integrate import quadrature


def test_simpsons_output() -> None:
    """Verify the calculation output from quadrature.simpson"""
    actual_integral = quadrature.simpson(lambda x: x**2, 0, 1, 100)
    expected_integral = 1 / 3
    assert abs(actual_integral - expected_integral) < 1e-6


def test_simpsons_error() -> None:
    """Verify that quadrature.simpson raises an exception for odd number of steps n"""
    with pytest.raises(ValueError) as exception:
        quadrature.simpson(lambda x: x, 0, 0, 1)
    assert str(exception.value) == "n must be even"
