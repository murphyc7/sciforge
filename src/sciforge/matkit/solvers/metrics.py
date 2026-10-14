"""Error metrics for comparing neural network predictions against reference solutions."""

import numpy as np


def _flatten_pair(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens two profiles to float64 vectors and checks they are comparable.

    Args:
        prediction (np.ndarray): Predicted profile, any shape that flattens to [N].
        reference (np.ndarray): Reference profile, any shape that flattens to [N].

    Returns:
        tuple[np.ndarray, np.ndarray]: The flattened prediction and reference.

    Raises:
        ValueError: If the two profiles do not contain the same number of points.
    """
    pred = np.asarray(prediction, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if pred.size != ref.size:
        raise ValueError(
            f"Prediction ({pred.size} points) and reference ({ref.size} points) differ in size."
        )
    return pred, ref


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    r"""Computes the relative L2 error :math:`\|u - u_{ref}\|_2 / \|u_{ref}\|_2`.

    Args:
        prediction (np.ndarray): Predicted profile.
        reference (np.ndarray): Reference profile on the same grid.

    Returns:
        float: The relative L2 error (0.0 for a perfect match).

    Raises:
        ValueError: If the profiles differ in size, or the reference is identically zero.
    """
    pred, ref = _flatten_pair(prediction, reference)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0.0:
        raise ValueError(
            "Relative error is undefined for an all-zero reference profile."
        )
    return float(np.linalg.norm(pred - ref)) / ref_norm


def max_abs_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Computes the maximum pointwise absolute error between two profiles.

    Args:
        prediction (np.ndarray): Predicted profile.
        reference (np.ndarray): Reference profile on the same grid.

    Returns:
        float: The largest absolute pointwise difference.

    Raises:
        ValueError: If the profiles differ in size.
    """
    pred, ref = _flatten_pair(prediction, reference)
    return float(np.max(np.abs(pred - ref)))
