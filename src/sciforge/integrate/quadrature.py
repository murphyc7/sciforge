"""murphyc7/sciforge/src/sciforge/integrate/quadrature.py"""

import numpy as np


def simpson(fnc: float, a: float, b: float, n: int) -> float:
    """Simpson's rule for numerical integration

    Args:
        fnc (float): function
        a (float): lower bound
        b (float): upper bound
        n (int): num steps

    Raises:
        ValueError: n must be even

    Returns:
        float: result of integration
    """
    # Validate input
    if n % 2 != 0:
        raise ValueError("n must be even")

    # Arrange quantities
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([fnc(xi) for xi in x])

    # Simpson's rule: h/3 * (y0 + 4*sum(odd) + 2*sum(even) + yn)
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])
    integral += 2 * np.sum(y[2:-1:2])
    integral *= h / 3

    return integral
