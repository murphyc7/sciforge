"""End-to-end accuracy benchmark of trained PINNs against classical reference solutions.

Trains a Carrier PINN with the production :class:`PINNTrainingWorker` loop and scores the
result against the matching reference from :mod:`sciforge.matkit.solvers.reference`. A low
training loss does not imply a correct solution, so this measures the error that matters:
the distance from the true solution of the governing equations.
"""

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    BasePhysicsSolver,
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
from sciforge.matkit.carrier_pinn.training_worker import (
    PINNTrainingWorker,
    TrainingMetrics,
)
from sciforge.matkit.solvers.metrics import relative_l2_error
from sciforge.matkit.solvers.reference import (
    ReferenceSolution,
    solve_constant_field,
    solve_coupled_poisson,
)

SUPPORTED_ENGINES = ("constant", "coupled")
COLLOCATION_POINTS = 100

# Boundary values mirror the problem setup used by the CLI and dashboard apps.
_CONSTANT_N_LEFT, _CONSTANT_N_RIGHT = 1.0, 0.0
_COUPLED_N_LEFT, _COUPLED_N_RIGHT = 1.0, 0.1
_COUPLED_PHI_LEFT, _COUPLED_PHI_RIGHT = 0.0, 0.5


@dataclass(frozen=True)
class BenchmarkResult:
    """Accuracy of one trained PINN against its classical reference solution.

    Attributes:
        engine_mode (str): Solver profile that was benchmarked ('constant' or 'coupled').
        epochs (int): Optimisation steps the PINN was trained for.
        seed (int): Random seed used for network initialisation.
        final_loss (float): Total training loss at the final epoch.
        n_rel_l2 (float): Relative L2 error of the carrier density against the reference.
        phi_rel_l2 (float | None): Relative L2 error of the potential, or None for engines
            that do not solve for it.
        elapsed_seconds (float): Wall-clock training time.
    """

    engine_mode: str
    epochs: int
    seed: int
    final_loss: float
    n_rel_l2: float
    phi_rel_l2: float | None
    elapsed_seconds: float


def run_benchmark(
    engine_mode: str,
    epochs: int = 300,
    seed: int = 0,
    effective_mass: float = 0.067,
    permittivity: float = 12.9,
) -> BenchmarkResult:
    """Trains a PINN and measures its error against the classical reference.

    Args:
        engine_mode (str): Either 'constant' or 'coupled'.
        epochs (int, optional): Optimisation steps. Defaults to 300.
        seed (int, optional): Random seed for network initialisation. Defaults to 0.
        effective_mass (float, optional): Effective mass relative to the electron mass.
            Defaults to 0.067 (GaAs).
        permittivity (float, optional): Relative static permittivity. Defaults to 12.9 (GaAs).

    Returns:
        BenchmarkResult: The accuracy summary for this run.

    Raises:
        ValueError: If ``engine_mode`` is not a supported engine.
        RuntimeError: If training aborts with an error.
    """
    if engine_mode not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported engine '{engine_mode}'. Choose from {SUPPORTED_ENGINES}."
        )

    torch.manual_seed(seed)
    x_interior = torch.linspace(0.0, 1.0, COLLOCATION_POINTS, dtype=torch.float32).view(
        -1, 1
    )
    x_left = torch.tensor([[0.0]], dtype=torch.float32)
    x_right = torch.tensor([[1.0]], dtype=torch.float32)

    physics_engine: BasePhysicsSolver
    if engine_mode == "constant":
        model = CarrierPINN(output_dim=1)
        physics_engine = ConstantFieldPhysics(
            effective_mass=effective_mass, permittivity=permittivity
        )
        n_left = torch.tensor([[_CONSTANT_N_LEFT]], dtype=torch.float32)
        n_right = torch.tensor([[_CONSTANT_N_RIGHT]], dtype=torch.float32)
    else:
        model = CarrierPINN(output_dim=2)
        physics_engine = PoissonCoupledPhysics(
            effective_mass=effective_mass, permittivity=permittivity
        )
        n_left = torch.tensor(
            [[_COUPLED_N_LEFT, _COUPLED_PHI_LEFT]], dtype=torch.float32
        )
        n_right = torch.tensor(
            [[_COUPLED_N_RIGHT, _COUPLED_PHI_RIGHT]], dtype=torch.float32
        )

    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=8)
    worker = PINNTrainingWorker(
        model=model,
        physics_engine=physics_engine,
        x_interior=x_interior,
        x_left=x_left,
        x_right=x_right,
        n_left=n_left,
        n_right=n_right,
        total_epochs=epochs,
        metrics_queue=metrics_queue,
        stop_event=threading.Event(),
        report_every=max(epochs, 1),
    )

    start_time = time.time()
    worker.run()
    elapsed = time.time() - start_time

    final: TrainingMetrics | None = None
    while not metrics_queue.empty():
        final = metrics_queue.get_nowait()
    if final is None or final.predictions is None:
        raise RuntimeError(
            f"Training did not produce predictions: {final.error if final else 'no output'}"
        )

    # The physics backend marks x_interior as requiring gradients, so detach before NumPy.
    grid = x_interior.detach().numpy().ravel().astype(np.float64)
    reference: ReferenceSolution
    if engine_mode == "constant":
        assert isinstance(physics_engine, ConstantFieldPhysics)
        reference = solve_constant_field(
            physics_engine, _CONSTANT_N_LEFT, _CONSTANT_N_RIGHT, x=grid
        )
    else:
        assert isinstance(physics_engine, PoissonCoupledPhysics)
        reference = solve_coupled_poisson(
            physics_engine,
            _COUPLED_N_LEFT,
            _COUPLED_N_RIGHT,
            _COUPLED_PHI_LEFT,
            _COUPLED_PHI_RIGHT,
            x=grid,
        )

    phi_error: float | None = None
    if reference.phi is not None:
        phi_error = relative_l2_error(final.predictions[:, 1], reference.phi)

    return BenchmarkResult(
        engine_mode=engine_mode,
        epochs=epochs,
        seed=seed,
        final_loss=final.total_loss,
        n_rel_l2=relative_l2_error(final.predictions[:, 0], reference.n),
        phi_rel_l2=phi_error,
        elapsed_seconds=elapsed,
    )


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Renders benchmark results as a GitHub-flavoured markdown table.

    Args:
        results (list[BenchmarkResult]): Results to tabulate, one row each.

    Returns:
        str: A markdown table with one row per result.
    """
    lines = [
        "| Engine | Epochs | Seed | Final loss | Rel. L2 error (n) | Rel. L2 error (phi) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        phi_cell = "n/a" if r.phi_rel_l2 is None else f"{r.phi_rel_l2:.2e}"
        lines.append(
            f"| {r.engine_mode} | {r.epochs} | {r.seed} | {r.final_loss:.2e} | "
            f"{r.n_rel_l2:.2e} | {phi_cell} |"
        )
    return "\n".join(lines)
