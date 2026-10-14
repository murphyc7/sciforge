"""Tests for murphyc7/sciforge/src/sciforge/matkit/solvers/benchmark.py"""

import numpy as np
import pytest
import torch

from sciforge.matkit.carrier_pinn.physics import PoissonCoupledPhysics
from sciforge.matkit.carrier_pinn.training_worker import TrainingMetrics
from sciforge.matkit.solvers import benchmark as benchmark_module
from sciforge.matkit.solvers.benchmark import (
    COLLOCATION_POINTS,
    BenchmarkResult,
    format_results_table,
    run_benchmark,
)
from sciforge.matkit.solvers.reference import solve_coupled_poisson

#############################################
# 1. Accuracy of trained PINNs vs reference #
#############################################


def test_constant_field_pinn_agrees_with_analytic_solution() -> None:
    """A short constant-field run must land within 1% of the exact analytic profile."""
    # Arrange / Act
    result = run_benchmark("constant", epochs=300, seed=0)

    # Assert
    assert result.n_rel_l2 < 1e-2
    assert result.phi_rel_l2 is None
    assert result.engine_mode == "constant"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Known issue: the 1e-3 Poisson loss weight lets the coupled PINN reach a low "
        "loss while remaining far from the true solution. Remove this marker when the "
        "loss weighting is fixed; strict mode then fails loudly if it is left behind."
    ),
)
def test_coupled_pinn_agrees_with_boundary_value_reference() -> None:
    """The coupled PINN should match the converged BVP reference to within 5%."""
    # Arrange / Act
    result = run_benchmark("coupled", epochs=300, seed=0)

    # Assert
    assert result.n_rel_l2 < 5e-2
    assert result.phi_rel_l2 is not None
    assert result.phi_rel_l2 < 5e-2


def test_coupled_benchmark_reports_both_fields() -> None:
    """The coupled benchmark must report finite errors for density and potential."""
    # Arrange / Act
    result = run_benchmark("coupled", epochs=20, seed=0)

    # Assert: Accuracy is not asserted here; this only checks the reporting contract
    assert result.engine_mode == "coupled"
    assert result.epochs == 20
    assert result.n_rel_l2 > 0.0
    assert result.phi_rel_l2 is not None and result.phi_rel_l2 > 0.0
    assert result.elapsed_seconds > 0.0


def test_benchmark_is_reproducible_for_a_fixed_seed() -> None:
    """Two runs with the same seed must produce identical errors."""
    # Arrange / Act
    first = run_benchmark("constant", epochs=10, seed=3)
    second = run_benchmark("constant", epochs=10, seed=3)

    # Assert
    assert first.n_rel_l2 == second.n_rel_l2
    assert first.final_loss == second.final_loss


def test_benchmark_rejects_unsupported_engine() -> None:
    """The bipolar solver has no reference yet and must be refused explicitly."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unsupported engine"):
        run_benchmark("bipolar", epochs=1)


def test_benchmark_surfaces_training_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that aborts must raise with its error message instead of scoring garbage."""

    # Arrange: Replace the worker with one that reports an aborted run
    class _AbortingWorker:
        def __init__(self, **kwargs: object) -> None:
            self.metrics_queue = kwargs["metrics_queue"]

        def run(self) -> None:
            self.metrics_queue.put_nowait(  # type: ignore[attr-defined]
                TrainingMetrics(
                    epoch=0,
                    total_loss=float("nan"),
                    bc_loss=float("nan"),
                    physics_loss=float("nan"),
                    elapsed_seconds=0.0,
                    is_final=True,
                    error="boom",
                )
            )

    monkeypatch.setattr(benchmark_module, "PINNTrainingWorker", _AbortingWorker)

    # Act / Assert
    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark("constant", epochs=1)


def test_benchmark_scores_each_field_against_its_own_reference_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known-answer test: scaling each reference field by a known factor must report that factor."""
    # Arrange: The reference on the benchmark's exact collocation grid
    grid = (
        torch.linspace(0.0, 1.0, COLLOCATION_POINTS, dtype=torch.float32)
        .numpy()
        .astype(np.float64)
    )
    reference = solve_coupled_poisson(
        PoissonCoupledPhysics(effective_mass=0.067, permittivity=12.9),
        n_left=1.0,
        n_right=0.1,
        phi_left=0.0,
        phi_right=0.5,
        x=grid,
    )
    assert reference.phi is not None
    density_offset, potential_offset = 0.01, 0.20
    predictions = np.stack(
        [
            reference.n * (1.0 + density_offset),
            reference.phi * (1.0 + potential_offset),
        ],
        axis=1,
    ).astype(np.float32)

    # Replace training with a worker that returns the scaled reference as its prediction
    class _ScriptedWorker:
        def __init__(self, **kwargs: object) -> None:
            self.metrics_queue = kwargs["metrics_queue"]

        def run(self) -> None:
            self.metrics_queue.put_nowait(  # type: ignore[attr-defined]
                TrainingMetrics(
                    epoch=1,
                    total_loss=0.5,
                    bc_loss=0.2,
                    physics_loss=0.3,
                    elapsed_seconds=0.1,
                    is_final=True,
                    predictions=predictions,
                )
            )

    monkeypatch.setattr(benchmark_module, "PINNTrainingWorker", _ScriptedWorker)

    # Act
    result = run_benchmark("coupled", epochs=1)

    # Assert: Swapping the density and potential channels would fail these checks
    assert result.n_rel_l2 == pytest.approx(density_offset, rel=1e-3)
    assert result.phi_rel_l2 == pytest.approx(potential_offset, rel=1e-3)


##############################
# 2. Results table rendering #
##############################


def test_format_results_table_renders_markdown_rows() -> None:
    """Each result becomes one row, with 'n/a' where no potential error exists."""
    # Arrange
    results = [
        BenchmarkResult("constant", 300, 0, 1.0e-4, 6.0e-4, None, 1.5),
        BenchmarkResult("coupled", 300, 1, 3.4e-2, 4.0e-1, 2.7e-1, 3.0),
    ]

    # Act
    table = format_results_table(results).splitlines()

    # Assert: Header, separator, then one row per result
    assert len(table) == 4
    assert table[0].startswith("| Engine")
    assert table[1].startswith("|---")
    assert "| constant | 300 | 0 |" in table[2]
    assert table[2].endswith("n/a |")
    assert "2.70e-01" in table[3]
