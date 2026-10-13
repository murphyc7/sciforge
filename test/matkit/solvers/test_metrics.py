"""Tests for murphyc7/sciforge/src/sciforge/matkit/solvers/metrics.py"""

import numpy as np
import pytest

from sciforge.matkit.solvers.metrics import max_abs_error, relative_l2_error


def test_relative_l2_error_is_zero_for_identical_profiles() -> None:
    """A perfect prediction must score exactly zero."""
    # Arrange
    profile = np.linspace(0.0, 1.0, 20)

    # Act / Assert
    assert relative_l2_error(profile, profile) == 0.0


def test_relative_l2_error_matches_hand_computed_value() -> None:
    """For reference [3, 4] (norm 5) and an error vector of norm 0.5, the result is 0.1."""
    # Arrange
    reference = np.array([3.0, 4.0])
    prediction = reference + np.array([0.3, 0.4])

    # Act / Assert
    assert relative_l2_error(prediction, reference) == pytest.approx(0.1)


def test_relative_l2_error_flattens_column_vectors() -> None:
    """PINN outputs are [N, 1] columns; they must compare cleanly against 1-D references."""
    # Arrange
    reference = np.array([1.0, 2.0, 3.0])
    prediction_column = (reference + 0.5).reshape(-1, 1)

    # Act / Assert
    assert relative_l2_error(prediction_column, reference) > 0.0


def test_relative_l2_error_rejects_size_mismatch_and_zero_reference() -> None:
    """Mismatched sizes and an all-zero reference are undefined and must raise."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="differ in size"):
        relative_l2_error(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="all-zero"):
        relative_l2_error(np.ones(3), np.zeros(3))


def test_max_abs_error_reports_the_largest_pointwise_deviation() -> None:
    """The maximum absolute error is the worst single-point miss."""
    # Arrange
    reference = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.1, 1.5, 3.0])

    # Act / Assert
    assert max_abs_error(prediction, reference) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="differ in size"):
        max_abs_error(np.zeros(2), np.zeros(3))
