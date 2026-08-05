"""Quadrature numerical integration techniques."""

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

    def __init__(self, f: Callable[[float | np.ndarray], float | np.ndarray]):
        """Initialise the integrator with a vectorised target function.

        Args:
            f (Callable): A vector-safe function accepting a float or a
                NumPy array and returning an equivalent shape.

        Raises:
            TypeError: If the provided target function 'f' is not a callable object.
        """
        if not callable(f):
            raise TypeError("The target function 'f' must be a callable object.")
        self.f = f
        self._eval_count = 0
        self.metrics: dict[str, Any] = {}

    def _eval_f(self, x: float | np.ndarray) -> np.ndarray:
        """Safely evaluate the target function and increment telemetry counters.

        Args:
            x (float | np.ndarray): Input coordinate or array of coordinates
                where the function should be evaluated.

        Returns:
            np.ndarray: Evaluated function values cast to a stable float64 array.

        Raises:
            ValueError: If the target function yields an infinite value, NaN element,
                or encounters a structural coordinate singularity.
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
        """Reset internal tracking metrics prior to running an integration technique.

        Args:
            None

        Returns:
            None
        """
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
        """Enforce strict domain boundaries validation on input integration limits.

        Args:
            a (float | np.ndarray): Lower bound(s) of the integration interval.
            b (float | np.ndarray): Upper bound(s) of the integration interval.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cleaned NumPy float64 tracking arrays
                representing the verified lower and upper bounds.

        Raises:
            ValueError: If the provided bounds contain NaN values or infinite parameters
                (unbounded calculations must use integrate_infinite instead).
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
    ) -> float | np.ndarray:
        """Evaluate the definite integral using the Composite Trapezoidal technique.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            n (int): Number of subintervals (panels) to divide the domain into.

        Returns:
            float | np.ndarray: A scalar float if input boundaries are scalars, or a
                NumPy array of matching dimension if vectorized arrays are provided.

        Raises:
            ValueError: If integration bounds contain illegal parameters, or if the
                number of subintervals 'n' is less than or equal to 0.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Subintervals 'n' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)
        max_ndim = max(a_arr.ndim, b_arr.ndim)

        if max_ndim > 0:
            t = np.linspace(0, 1, n + 1).reshape(-1, *(1 for _ in range(max_ndim)))
        else:
            t = np.linspace(0, 1, n + 1)

        x = a_arr + t * (b_arr - a_arr)
        y = self._eval_f(x)
        h = (b_arr - a_arr) / n

        # y[0] to add the scalar boundary correctly
        integral = (h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1], axis=0) + y[-1])

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return (
            integral.item()
            if (hasattr(integral, "ndim") and integral.ndim == 0)
            else integral
        )

    def simpsons(
        self, a: float | np.ndarray, b: float | np.ndarray, n: int
    ) -> float | np.ndarray:
        """Evaluate the definite integral using the Composite Simpson's 1/3 technique.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            n (int): Number of subintervals (must be a positive even integer).

        Returns:
            float | np.ndarray: A scalar float if input boundaries are scalars, or a
                NumPy array of matching dimension if vectorized arrays are provided.

        Raises:
            ValueError: If boundaries fail validation checks, or if subintervals 'n'
                is not a positive even integer value.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0 or n % 2 != 0:
            raise ValueError("Simpson's rule requires a positive even integer for 'n'.")

        a_arr, b_arr = self._validate_bounds(a, b)
        max_ndim = max(a_arr.ndim, b_arr.ndim)

        if max_ndim > 0:
            t = np.linspace(0, 1, n + 1).reshape(-1, *(1 for _ in range(max_ndim)))
        else:
            t = np.linspace(0, 1, n + 1)

        x = a_arr + t * (b_arr - a_arr)
        y = self._eval_f(x)
        h = (b_arr - a_arr) / n

        # y[0] to ensure correct numeric shape matching
        integral = (h / 3.0) * (
            y[0]
            + 4.0 * np.sum(y[1:-1:2], axis=0)
            + 2.0 * np.sum(y[2:-1:2], axis=0)
            + y[-1]
        )

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return (
            integral.item()
            if (hasattr(integral, "ndim") and integral.ndim == 0)
            else integral
        )

    def gauss_legendre(
        self, a: float | np.ndarray, b: float | np.ndarray, deg: int
    ) -> float | np.ndarray:
        """Evaluate the definite integral using Gauss-Legendre Quadrature.

        Calculates optimal nodes and weights dynamically via the Golub-Welsch symmetric tridiagonal
        matrix eigenvalue solver. Exact for polynomials of degree 2N - 1 or less.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            deg (int): Polynomial degree specifying the number of quadrature nodes (N).

        Returns:
            float | np.ndarray: Calculated numerical integral approximation. Returns a scalar
                float for scalar inputs, or a NumPy array if evaluating vector bounds.

        Raises:
            ValueError: If polynomial degree 'deg' is not a strictly positive integer.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(deg, (int, np.integer)) or deg <= 0:
            raise ValueError("Polynomial degree 'deg' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)
        max_ndim = max(a_arr.ndim, b_arr.ndim)

        i = np.arange(1, deg)
        beta = i / np.sqrt(4.0 * i**2 - 1.0)
        T = np.diag(beta, 1) + np.diag(beta, -1)  # noqa: N806

        nodes, evecs = np.linalg.eigh(T)
        weights = 2.0 * (evecs[0, :] ** 2)

        if max_ndim > 0:
            shape_modifier = (-1, *(1 for _ in range(max_ndim)))
            nodes_grid = nodes.reshape(shape_modifier)
            weights_grid = weights.reshape(shape_modifier)
            x = 0.5 * (b_arr - a_arr) * nodes_grid + 0.5 * (b_arr + a_arr)
            y = self._eval_f(x)
            integral = 0.5 * (b_arr - a_arr) * np.sum(weights_grid * y, axis=0)
        else:
            x = 0.5 * (b_arr - a_arr) * nodes + 0.5 * (b_arr + a_arr)
            y = self._eval_f(x)
            integral = 0.5 * (b_arr - a_arr) * np.sum(weights * y)

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return (
            integral.item()
            if (hasattr(integral, "ndim") and integral.ndim == 0)
            else integral
        )

    def adaptive_simpsons(
        self, a: float, b: float, tol: float = 1e-6, max_depth: int = 50
    ) -> float:
        """Evaluate the integral using a recursive Adaptive Simpson's technique.

        Dynamically refines structural grid steps across specific sub-domains
        where the target function experiences rapid spatial changes or high oscillations.

        Args:
            a (float): Lower bound of the integration domain.
            b (float): Upper bound of the integration domain.
            tol (float): Local accuracy error tolerance threshold. Defaults to 1e-6.
            max_depth (int): Maximum recursion depth ceiling to protect against infinite
                subdivision loops. Defaults to 50.

        Returns:
            float: The highly refined numerical integration solution value.

        Raises:
            ValueError: If input bounds are NaN/Inf, or if error tolerance 'tol' or
                'max_depth' are set to zero or a negative value.
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
        """Internal recursive calculation step managing error budgeting and interval segmentation.

        Args:
            a (float): Lower boundary coordinate of the targeted segment.
            b (float): Upper boundary coordinate of the targeted segment.
            tol (float): Segment error tolerance budget restriction.
            s_whole (float): Coarse Simpson value over the total subinterval region.
            fa (float): Cached function evaluation at lower bound a.
            fc (float): Cached function evaluation at midpoint c.
            fb (float): Cached function evaluation at upper bound b.
            max_depth (int): Maximum depth configuration cap limit.
            depth (int): Present structural iteration depth location index tracker.
            state (dict): Reference tracking state dictionary holding telemetry accumulators.

        Returns:
            float: Evaluated segment approximation adjusted with explicit Richardson Extrapolation.
        """
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
    ) -> float | np.ndarray:
        """Evaluate the definite integral using the Clenshaw-Curtis technique.

        Args:
            a (float | np.ndarray): Lower integration limit(s).
            b (float | np.ndarray): Upper integration limit(s).
            n (int): Number of nodes representing polynomial interpolation depth order.

        Returns:
            float | np.ndarray: Mapped integration output array matching input dimension configurations.

        Raises:
            ValueError: If node resolution value 'n' is not a strictly positive integer.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Node limit 'n' must be a positive integer.")

        a_arr, b_arr = self._validate_bounds(a, b)
        max_ndim = max(a_arr.ndim, b_arr.ndim)

        theta = np.pi * np.arange(n + 1) / n
        nodes = np.cos(theta)

        weights = np.zeros(n + 1)
        v = np.ones(n - 1)
        for k in range(2, n, 2):
            v -= 2.0 * np.cos(k * theta[1:-1]) / (k**2 - 1.0)
        if n % 2 == 0:
            v -= np.cos(n * theta[1:-1]) / (n**2 - 1.0)

        weights[1:-1] = 2.0 * v / n
        endpoint_val = 1.0 / (n**2 - 1.0) if n % 2 == 0 else 1.0 / n**2

        # Target individual indices to preserve array mutations
        weights[0] = endpoint_val
        weights[-1] = endpoint_val

        if max_ndim > 0:
            shape_modifier = (-1, *(1 for _ in range(max_ndim)))
            nodes_grid = nodes.reshape(shape_modifier)
            weights_grid = weights.reshape(shape_modifier)
            x = 0.5 * (b_arr - a_arr) * nodes_grid + 0.5 * (b_arr + a_arr)
            y = self._eval_f(x)
            integral = 0.5 * (b_arr - a_arr) * np.sum(weights_grid * y, axis=0)
        else:
            x = 0.5 * (b_arr - a_arr) * nodes + 0.5 * (b_arr + a_arr)
            y = self._eval_f(x)
            integral = 0.5 * (b_arr - a_arr) * np.sum(weights * y)

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return (
            integral.item()
            if (hasattr(integral, "ndim") and integral.ndim == 0)
            else integral
        )

    def integrate_infinite(
        self, a: float, b: float, method: str = "gauss_legendre", **kwargs
    ) -> float | np.ndarray:
        """Evaluate improper integrals over semi-infinite or infinite domain fields.

        Leverages adaptive change-of-variable coordinate mappings to condense unbounded endpoints into
        a closed, finite integration landscape smoothly.

        Args:
            a (float): Lower integration boundary limit (can set explicitly to -np.inf).
            b (float): Upper integration boundary limit (can set explicitly to np.inf).
            method (str): Name string selector referencing target execution technique. Defaults to "gauss_legendre".
            **kwargs: Configuration flags and hyper-parameters passed downstream to backend runners.

        Returns:
            float | np.ndarray: Evaluated numerical result over the mapped continuous domain.

        Raises:
            ValueError: If both bounds are finite numbers, or range sequencing directions are mismatched.
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
            transformed_f = lambda t: (  # noqa: E731
                original_f(t / (1.0 - t**2)) * (1.0 + t**2) / (1.0 - t**2) ** 2
            )
            target_a, target_b = -1.0 + 1e-15, 1.0 - 1e-15
        elif not inf_a and inf_b:
            transformed_f = lambda t: (  # noqa: E731
                original_f(a + t / (1.0 - t)) * (1.0 / (1.0 - t) ** 2)
            )
            target_a, target_b = 0.0, 1.0 - 1e-15
        else:
            transformed_f = lambda t: original_f(b - (1.0 - t) / t) * (1.0 / t**2)  # noqa: E731
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
        return res

    def tanh_sinh(
        self, a: float, b: float, tol: float = 1e-9, max_level: int = 6
    ) -> float:
        """Evaluate functions using the Tanh-Sinh Double Exponential quadrature technique.

        Args:
            a (float): Lower bounding tracking index coordinate.
            b (float): Upper bounding tracking index coordinate.
            tol (float): Targeted precision error budget threshold convergence floor. Defaults to 1e-9.
            max_level (int): Maximum geometric loop generation slicing depth steps. Defaults to 6.

        Returns:
            float: Highly resolved numerical evaluation calculation output.

        Raises:
            ValueError: If accuracy convergence budget parameters evaluate below zero thresholds.
        """
        start_time = time.perf_counter()
        self._reset_telemetry()

        if tol <= 0:
            raise ValueError("Tolerance targets must evaluate positively.")

        scale_diff = 0.5 * (b - a)
        scale_sum = 0.5 * (b + a)
        previous_integral = 0.0
        integral = 0.0

        for level in range(1, max_level + 1):
            h = 0.5**level
            # Generate points spanning outward across the functional landscape mapping limit
            t = np.arange(h, 6.0, h)

            with np.errstate(over="ignore", invalid="ignore"):
                sinh_t = np.sinh(t)
                cosh_t = np.cosh(t)
                arg = 0.5 * np.pi * sinh_t

                valid_mask = arg < 200
                if not np.any(valid_mask):
                    continue

                t = t[valid_mask]
                sinh_t = sinh_t[valid_mask]
                cosh_t = cosh_t[valid_mask]
                arg = arg[valid_mask]

                tanh_arg = np.tanh(arg)
                w = scale_diff * (0.5 * np.pi * cosh_t) / (np.cosh(arg) ** 2)

            x_pts_pos = scale_sum + scale_diff * tanh_arg
            x_pts_neg = scale_sum - scale_diff * tanh_arg

            # Base midpoint evaluation at t=0
            total_sum = float(self._eval_f(scale_sum)) * (scale_diff * 0.5 * np.pi)

            # Enforce calculations strictly inside the open boundary constraints
            pos_mask = (x_pts_pos > a) & (x_pts_pos < b) & np.isfinite(w) & (w > 1e-15)
            neg_mask = (x_pts_neg > a) & (x_pts_neg < b) & np.isfinite(w) & (w > 1e-15)

            if np.any(pos_mask):
                total_sum += np.sum(self._eval_f(x_pts_pos[pos_mask]) * w[pos_mask])
            if np.any(neg_mask):
                total_sum += np.sum(self._eval_f(x_pts_neg[neg_mask]) * w[neg_mask])

            integral = h * total_sum

            if level > 2 and abs(integral - previous_integral) < tol:
                break
            previous_integral = integral

        self.metrics["function_evaluations"] = self._eval_count
        self.metrics["execution_time_seconds"] = time.perf_counter() - start_time
        return float(integral)
