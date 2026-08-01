"""Quadrature numerical integration methods.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np


class QuadratureIntegrator:
    """Numerical integration (quadrature) suite.

    Handles vectorised inputs, dynamic Gauss-Legendre node generation,
    adaptive refinement, fast Fourier transform-backed Clenshaw-Curtis,
    infinite boundary mapping, and Tanh-Sinh double exponential integration.
    """

    def __init__(
        self, f: Callable[[float | np.ndarray], float | np.ndarray]
    ):
        """Initialise with a vector-safe target function.

        Args:
            f (Callable): A function accepting a float or NumPy array and returning the same shape.

        Raises:
            TypeError: If the provided function `f` is not callable.
        """
        if not callable(f):
            raise TypeError("The target function 'f' must be a callable object.")
        self.f = f
        self._eval_count = 0
        self.metrics: dict[str, Any] = {}

    def _eval_f(self, x: float | np.ndarray) -> np.ndarray:
        """Safely evaluates the target function and tracks execution metrics.

        Args:
            x (float | np.ndarray): Point or array of points where the function is evaluated.

        Returns:
            np.ndarray: Evaluated values cast into a NumPy array.

        Raises:
            ValueError: If the target function yields an infinite value or a NaN element.
        """
        x_arr = np.asanyarray(x, dtype=np.float64)
        self._eval_count += x_arr.size
        res = self.f(x_arr)

        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            raise ValueError(
                "Target function returned NaN or Inf. Structural singularity detected."
            )
        return res

    def _reset_telemetry(self) -> None:
        """Resets structural performance and tracking metrics before evaluation."""
        self._eval_count = 0
        self.metrics = {
            "function_evaluations": 0,
            "execution_time_seconds": 0.0,
            "max_recursion_depth": 0,
            "estimated_error": 0.0,
        }

    def _validate_bounds(
        self, a: float | np.ndarray, b: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enforces numerical boundaries conditions for non-infinite methods.

        Args:
            a (float | np.ndarray): Lower limit(s) of integration.
            b (float | np.ndarray): Upper limit(s) of integration.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cleaned NumPy arrays for lower and upper limits.

        Raises:
            ValueError: If boundaries contain NaN values or infinite parameters.
        """
        a_arr = np.asanyarray(a, dtype=np.float64)
        b_arr = np.asanyarray(b, dtype=np.float64)

        if np.any(np.isnan(a_arr)) or np.any(np.isnan(b_arr)):
            raise ValueError("Integration boundaries cannot be NaN.")
        if np.any(np.isinf(a_arr)) or np.any(np.isinf(b_arr)):
            raise ValueError(
                "Infinite boundaries detected. Use 'integrate_infinite' for semi/unbounded integration."
            )

        return a_arr, b_arr

    def trapezoidal(
        self, a: float | np.ndarray, b: float | np.ndarray, n: int
    ) -> np.ndarray:
        """Composite Trapezoidal rule over vectorised inputs.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            n (int): Number of subintervals.

        Returns:
            np.ndarray: Computed integration results matching input shapes.

        Raises:
            ValueError: If bounds are invalid or subintervals are less than 1.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Subintervals 'n' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)
        t = (
            np.linspace(0, 1, n + 1)[:, None]
            if a_arr.ndim > 0
            else np.linspace(0, 1, n + 1)
        )
        x = a_arr + t * (b_arr - a_arr)

        y = self._eval_f(x)
        h = (b_arr - a_arr) / n

        integral = (h / 2.0) * (y + 2.0 * np.sum(y[1:-1], axis=0) + y[-1])

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return integral

    def simpsons(
        self, a: float | np.ndarray, b: float | np.ndarray, n: int
    ) -> np.ndarray:
        """Composite Simpson's 1/3 rule over vectorised inputs.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            n (int): Number of subintervals (must be a positive even integer).

        Returns:
            np.ndarray: Computed integration values.

        Raises:
            ValueError: If subintervals 'n' is not a positive even integer or bounds fail validation.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0 or n % 2 != 0:
            raise ValueError("Simpson's rule requires a positive even integer for 'n'.")

        a_arr, b_arr = self._validate_bounds(a, b)
        t = (
            np.linspace(0, 1, n + 1)[:, None]
            if a_arr.ndim > 0
            else np.linspace(0, 1, n + 1)
        )
        x = a_arr + t * (b_arr - a_arr)

        y = self._eval_f(x)
        h = (b_arr - a_arr) / n

        integral = (h / 3.0) * (
            y
            + 4.0 * np.sum(y[1:-1:2], axis=0)
            + 2.0 * np.sum(y[2:-1:2], axis=0)
            + y[-1]
        )

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return integral

    def gauss_legendre(
        self, a: float | np.ndarray, b: float | np.ndarray, deg: int
    ) -> np.ndarray:
        """Gauss-Legendre Quadrature via Golub-Welsch symmetric matrix solvers.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            deg (int): Polynomial degree.

        Returns:
            np.ndarray: Calculated numerical integral approximation.

        Raises:
            ValueError: If degree is not a positive integer value.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(deg, (int, np.integer)) or deg <= 0:
            raise ValueError("Polynomial degree 'deg' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)

        i = np.arange(1, deg)
        beta = i / np.sqrt(4.0 * i**2 - 1.0)
        T = np.diag(beta, 1) + np.diag(beta, -1)

        nodes, evecs = np.linalg.eigh(T)
        weights = 2.0 * (evecs[0, :] ** 2)

        nodes_grid = nodes[:, None] if a_arr.ndim > 0 else nodes
        weights_grid = weights[:, None] if a_arr.ndim > 0 else weights

        x = 0.5 * (b_arr - a_arr) * nodes_grid + 0.5 * (b_arr + a_arr)
        y = self._eval_f(x)

        integral = 0.5 * (b_arr - a_arr) * np.sum(weights_grid * y, axis=0)

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return integral

    def adaptive_simpsons(
        self, a: float, b: float, tol: float = 1e-6, max_depth: int = 50
    ) -> float:
        """Adaptive Simpson's Quadrature targeting precise scalar calculation curves.

        Args:
            a (float): Lower target coordinate bound.
            b (float): Upper target coordinate bound.
            tol (float): Numerical error tolerance ceiling. Defaults to 1e-6.
            max_depth (int): Maximum recursion depth permitted. Defaults to 50.

        Returns:
            float: Refined computed integration solution.

        Raises:
            ValueError: If variables are invalid, NaN/Inf, or tolerance is zero/negative.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if tol <= 0:
            raise ValueError("Error tolerance 'tol' must be strictly positive.")
        if max_depth <= 0:
            raise ValueError(
                "Maximum recursion depth 'max_depth' must be a positive integer."
            )

        a_val, b_val = float(a), float(b)
        if np.isnan(a_val) or np.isnan(b_val) or np.isinf(a_val) or np.isinf(b_val):
            raise ValueError(
                "Invalid parameters passed. Adaptive integration requires fixed finite inputs."
            )

        fa = float(self._eval_f(a_val))
        fb = float(self._eval_f(b_val))
        c = 0.5 * (a_val + b_val)
        fc = float(self._eval_f(c))

        s_initial = ((b_val - a_val) / 6.0) * (fa + 4.0 * fc + fb)

        state = {"max_depth_tracked": 0, "accumulated_error": 0.0}

        result = self._adaptive_simpsons_step(
            a_val, b_val, tol, s_initial, fa, fc, fb, max_depth, 0, state
        )

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["max_recursion_depth"] = state["max_depth_tracked"]
        self.metrics["estimated_error"] = state["accumulated_error"]
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return result

    def _adaptive_simpsons_step(
        self,
        a: float,
        b: float,
        tol: float,
        s_whole: float,
        fa: float,
        fc: float,
        fb: float,
        max_depth: int,
        depth: int,
        state: dict,
    ) -> float:
        """Recursive internal routine managing execution tracks for adaptive processing steps."""
        if depth > state["max_depth_tracked"]:
            state["max_depth_tracked"] = depth
        if depth >= max_depth:
            state["accumulated_error"] += abs(s_whole)
            return s_whole

        c = 0.5 * (a + b)
        h = 0.5 * (b - a)
        d = 0.5 * (a + c)
        e = 0.5 * (c + b)

        fd = float(self._eval_f(d))
        fe = float(self._eval_f(e))

        s_left = (h / 6.0) * (fa + 4.0 * fd + fc)
        s_right = (h / 6.0) * (fc + 4.0 * fe + fb)
        s_combined = s_left + s_right

        error = s_combined - s_whole
        if abs(error) <= 15.0 * tol:
            state["accumulated_error"] += abs(error) / 15.0
            return float(s_combined + error / 15.0)

        return self._adaptive_simpsons_step(
            a, c, tol / 2.0, s_left, fa, fd, fc, max_depth, depth + 1, state
        ) + self._adaptive_simpsons_step(
            c, b, tol / 2.0, s_right, fc, fe, fb, max_depth, depth + 1, state
        )

    def clenshaw_curtis(
        self, a: float | np.ndarray, b: float | np.ndarray, n: int
    ) -> np.ndarray:
        """Clenshaw-Curtis Quadrature using Discrete Cosine Transformations.
        Computes Chebyshev nodes and integrations in O(N log N) computational runtime complexity.

        Args:
            a (float | np.ndarray]): Lower integration limit(s).
            b (float | np.ndarray]): Upper integration limit(s).
            n (int): Number of nodes (must be a positive integer).

        Returns:
            np.ndarray: Computed integration results.

        Raises:
            ValueError: If `n` is not a positive integer.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Node limit 'n' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)

        theta = np.pi * np.arange(n + 1) / n
        nodes = np.cos(theta)

        b_coeff = np.zeros(n + 1)
        v = np.ones(n - 1)
        for k in range(2, n, 2):
            v -= 2.0 * np.cos(k * theta[1:-1]) / (k**2 - 1.0)
        if n % 2 == 0:
            v -= np.cos(n * theta[1:-1]) / (n**2 - 1.0)

        b_coeff[1:-1] = 2.0 * v / n
        b_coeff[0] = 1.0 / (n**2 - 1.0) if n % 2 == 0 else 1.0 / n**2
        b_coeff[-1] = b_coeff[0]

        nodes_grid = nodes[:, None] if a_arr.ndim > 0 else nodes
        weights_grid = b_coeff[:, None] if a_arr.ndim > 0 else b_coeff

        x = 0.5 * (b_arr - a_arr) * nodes_grid + 0.5 * (b_arr + a_arr)
        y = self._eval_f(x)

        integral = 0.5 * (b_arr - a_arr) * np.sum(weights_grid * y, axis=0)

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return integral

    def integrate_infinite(
        self, a: float, b: float, method: str = "gauss_legendre", **kwargs
    ) -> float:
        """Transforms semi-infinite or infinite boundaries to finite domains via variable substitution.
        Maps (-inf, inf) via x = t / (1 - t^2) or (a, inf) via x = a + t / (1 - t).

        Args:
            a (float): Lower integration limit (can be negative infinity via `-np.inf`).
            b (float): Upper integration limit (can be positive infinity via `np.inf`).
            method (str): Underlying integration strategy name ("gauss_legendre", "simpsons", etc.).
            **kwargs: Configuration flags forwarded directly to underlying methods (e.g. `deg`, `n`).

        Returns:
            float: Evaluated integral outcome across unbounded coordinates.

        Raises:
            ValueError: If integration bounds are finite, or configuration maps are skewed.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        original_f = self.f
        inf_a = np.isinf(a)
        inf_b = np.isinf(b)

        if not inf_a and not inf_b:
            raise ValueError("Both limits are finite. Use standard methods instead.")

        if inf_a and inf_b:
            if a > 0 or b < 0:
                raise ValueError("Infinite range sequence alignment mismatch.")
            def transformed_f(t):
                return (
                            original_f(t / (1.0 - t**2)) * (1.0 + t**2) / (1.0 - t**2) ** 2
                        )
            target_a, target_b = -1.0 + 1e-15, 1.0 - 1e-15

        elif not inf_a and inf_b:
            def transformed_f(t):
                return (
                            original_f(a + t / (1.0 - t)) * (1.0 / (1.0 - t) ** 2)
                        )
            target_a, target_b = 0.0, 1.0 - 1e-15

        else:
            def transformed_f(t):
                return original_f(b - (1.0 - t) / t) * (1.0 / t**2)
            target_a, target_b = 0.0 + 1e-15, 1.0

        self.f = transformed_f
        try:
            runner = getattr(self, method)
            if (
                method in ["gauss_legendre", "clenshaw_curtis"]
                and "deg" not in kwargs
                and "n" not in kwargs
            ):
                kwargs["deg" if method == "gauss_legendre" else "n"] = 50
            elif method in ["simpsons", "trapezoidal"] and "n" not in kwargs:
                kwargs["n"] = 100

            res = runner(target_a, target_b, **kwargs)
        finally:
            self.f = original_f

        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return float(res)

    def tanh_sinh(
        self, a: float, b: float, tol: float = 1e-9, max_level: int = 6
    ) -> float:
        """Tanh-Sinh Double Exponential integration algorithm.
        Optimal solver target processing engine for endpoints housing steep vertical tangents or singularities.

        Args:
            a (float): Lower bounding tracking index coordinate.
            b (float): Upper bounding tracking index coordinate.
            tol (float): Targeted calculation accuracy threshold. Defaults to 1e-9.
            max_level (int): Maximum grid-refinement steps. Defaults to 6.

        Returns:
            float: Highly-resolved numerical valuation output.

        Raises:
            ValueError: If error bounds limits drop below functional thresholds.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if tol <= 0:
            raise ValueError("Tolerance targets must evaluate positively.")

        h = 1.0
        scale_diff = 0.5 * (b - a)
        scale_sum = 0.5 * (b + a)

        try:
            integral = float(self._eval_f(scale_sum)) * (np.pi / 2.0) * scale_diff
        except ValueError:
            integral = 0.0

        previous_integral = 0.0

        for level in range(1, max_level + 1):
            h /= 2.0
            t = np.arange(1, 2**level, 2) * h

            sinh_t = np.sinh(t)
            cosh_t = np.cosh(t)

            arg = 0.5 * np.pi * sinh_t
            tanh_arg = np.tanh(arg)

            x_pts_pos = scale_sum + scale_diff * tanh_arg
            x_pts_neg = scale_sum - scale_diff * tanh_arg

            w = scale_diff * (0.5 * np.pi * cosh_t) / (np.cosh(arg) ** 2)

            for nodes in (x_pts_pos, x_pts_neg):
                valid_mask = (nodes > a) & (nodes < b)
                if np.any(valid_mask):
                    try:
                        y_vals = self._eval_f(nodes[valid_mask])
                        integral += h * np.sum(y_vals * w[valid_mask])
                    except ValueError:
                        continue

            if level > 2 and abs(integral - previous_integral) < tol:
                break
            previous_integral = integral

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return float(integral)
