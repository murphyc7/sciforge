"""Tests for murphyc7/sciforge/src/sciforge/matkit/carrier_pinn/training_worker.py"""

import math
import queue
import threading

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

###############################################
# 0. Shared builders and physics test doubles #
###############################################


class _FailingPhysics(BasePhysicsSolver):
    """Physics double that raises on the first residual evaluation."""

    def compute_residuals(
        self, x: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError("boom")


class _CancellingPhysics(ConstantFieldPhysics):
    """Real constant-field physics that requests cancellation after N evaluations."""

    def __init__(self, stop_event: threading.Event, cancel_after: int) -> None:
        super().__init__(effective_mass=0.067, permittivity=12.9)
        self.stop_event = stop_event
        self.cancel_after = cancel_after
        self.calls = 0

    def compute_residuals(
        self, x_scaled: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        if self.calls >= self.cancel_after:
            self.stop_event.set()
        return super().compute_residuals(x_scaled, model)


def _build_worker(
    total_epochs: int,
    physics_engine: BasePhysicsSolver | None = None,
    metrics_queue: "queue.Queue[TrainingMetrics] | None" = None,
    stop_event: threading.Event | None = None,
    report_every: int = 5,
    seed: int = 0,
) -> PINNTrainingWorker:
    """Builds a small, seeded worker around the constant-field problem."""
    torch.manual_seed(seed)
    return PINNTrainingWorker(
        model=CarrierPINN(input_dim=1, output_dim=1, hidden_dim=16),
        physics_engine=physics_engine or ConstantFieldPhysics(0.067, 12.9),
        x_interior=torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(-1, 1),
        x_left=torch.tensor([[0.0]], dtype=torch.float32),
        x_right=torch.tensor([[1.0]], dtype=torch.float32),
        n_left=torch.tensor([[1.0]], dtype=torch.float32),
        n_right=torch.tensor([[0.0]], dtype=torch.float32),
        total_epochs=total_epochs,
        metrics_queue=metrics_queue or queue.Queue(maxsize=200),
        stop_event=stop_event or threading.Event(),
        report_every=report_every,
    )


def _drain(metrics_queue: "queue.Queue[TrainingMetrics]") -> list[TrainingMetrics]:
    """Empties a metrics queue non-blockingly, preserving arrival order."""
    snapshots: list[TrainingMetrics] = []
    while True:
        try:
            snapshots.append(metrics_queue.get_nowait())
        except queue.Empty:
            return snapshots


#####################################
# 1. Metrics streaming and contract #
#####################################


def test_worker_streams_ordered_metrics_and_one_terminal_snapshot() -> None:
    """Verifies progress snapshots arrive in order and exactly one terminal snapshot closes the run."""
    # Arrange: A short run reporting every 5 epochs, executed synchronously
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = _build_worker(total_epochs=10, metrics_queue=metrics_queue)

    # Act
    worker.run()
    snapshots = _drain(metrics_queue)

    # Assert: Epochs never go backwards and wall-clock time never decreases
    epochs = [s.epoch for s in snapshots]
    assert epochs == sorted(epochs)
    assert [s.elapsed_seconds for s in snapshots] == sorted(
        s.elapsed_seconds for s in snapshots
    )

    # Assert: The first report is epoch 0 and only the last snapshot is terminal
    assert epochs[0] == 0
    assert [s.is_final for s in snapshots] == [False] * (len(snapshots) - 1) + [True]


def test_worker_terminal_snapshot_carries_predictions_over_the_interior_grid() -> None:
    """Confirms the terminal snapshot hands back evaluation-mode predictions and no error."""
    # Arrange
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = _build_worker(total_epochs=5, metrics_queue=metrics_queue)

    # Act
    worker.run()
    final = _drain(metrics_queue)[-1]

    # Assert
    assert final.is_final
    assert final.error is None
    assert final.predictions is not None
    assert final.predictions.shape == (16, 1)
    assert math.isfinite(final.total_loss)


def test_worker_optimisation_reduces_total_loss() -> None:
    """Guards the optimisation loop itself: the reported loss must fall over a short run."""
    # Arrange
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=500)
    worker = _build_worker(
        total_epochs=150, metrics_queue=metrics_queue, report_every=10
    )

    # Act
    worker.run()
    snapshots = _drain(metrics_queue)

    # Assert: Compare the first progress report against the terminal snapshot
    assert snapshots[-1].total_loss < snapshots[0].total_loss


def test_worker_clamps_report_interval_to_at_least_one_epoch() -> None:
    """Ensures a zero or negative reporting interval cannot cause a modulo-by-zero."""
    # Arrange / Act
    worker = _build_worker(total_epochs=1, report_every=0)

    # Assert
    assert worker.report_every == 1


################################
# 2. Cancellation and failures #
################################


def test_worker_cancels_cooperatively_and_still_emits_terminal_snapshot() -> None:
    """Verifies stop_event ends the loop early while preserving the single terminal snapshot."""
    # Arrange: Physics double that requests a stop during the third epoch
    stop_event = threading.Event()
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = _build_worker(
        total_epochs=50,
        physics_engine=_CancellingPhysics(stop_event, cancel_after=3),
        metrics_queue=metrics_queue,
        stop_event=stop_event,
    )

    # Act
    worker.run()
    snapshots = _drain(metrics_queue)

    # Assert: Ended well before 50 epochs, with a clean terminal snapshot
    final = snapshots[-1]
    assert final.is_final
    assert final.epoch < 50
    assert final.error is None
    assert final.predictions is not None
    assert sum(s.is_final for s in snapshots) == 1


def test_worker_reports_exceptions_through_the_terminal_snapshot() -> None:
    """Verifies a failing physics backend is surfaced as data rather than a dead thread."""
    # Arrange
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = _build_worker(
        total_epochs=5, physics_engine=_FailingPhysics(), metrics_queue=metrics_queue
    )

    # Act: run() must not raise; the failure travels through the queue
    worker.run()
    snapshots = _drain(metrics_queue)

    # Assert
    assert len(snapshots) == 1
    final = snapshots[0]
    assert final.is_final
    assert final.error == "boom"
    assert final.predictions is None
    assert math.isnan(final.total_loss)


############################
# 3. Queue and concurrency #
############################


def test_worker_bounded_queue_drops_oldest_but_keeps_terminal_snapshot() -> None:
    """Confirms a slow consumer never blocks training and always sees the final result."""
    # Arrange: A tiny queue with a report every epoch guarantees overflow
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=2)
    worker = _build_worker(total_epochs=6, metrics_queue=metrics_queue, report_every=1)

    # Act
    worker.run()
    snapshots = _drain(metrics_queue)

    # Assert: Only the newest entries survive, and the last one is the terminal snapshot
    assert len(snapshots) == 2
    assert snapshots[-1].is_final


def test_worker_runs_to_completion_on_a_background_thread() -> None:
    """Exercises the real concurrency contract: run() as a Thread target feeding the queue."""
    # Arrange
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = _build_worker(total_epochs=10, metrics_queue=metrics_queue)
    thread = threading.Thread(target=worker.run, daemon=True)

    # Act
    thread.start()
    thread.join(timeout=60)

    # Assert: The thread finished and the consumer side sees a terminal snapshot
    assert not thread.is_alive()
    snapshots = _drain(metrics_queue)
    assert snapshots[-1].is_final
    assert snapshots[-1].error is None


##########################################
# 4. Coupled engine and queue race cases #
##########################################


def test_worker_trains_coupled_poisson_engine_end_to_end() -> None:
    """Exercises the Poisson-scaled loss branch with a two-channel [n, phi] network."""
    # Arrange: Coupled boundary targets map [n, phi] at each end of the domain
    torch.manual_seed(0)
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = PINNTrainingWorker(
        model=CarrierPINN(input_dim=1, output_dim=2, hidden_dim=16),
        physics_engine=PoissonCoupledPhysics(effective_mass=0.067, permittivity=12.9),
        x_interior=torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(-1, 1),
        x_left=torch.tensor([[0.0]], dtype=torch.float32),
        x_right=torch.tensor([[1.0]], dtype=torch.float32),
        n_left=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        n_right=torch.tensor([[0.1, 0.5]], dtype=torch.float32),
        total_epochs=5,
        metrics_queue=metrics_queue,
        stop_event=threading.Event(),
    )

    # Act
    worker.run()
    final = _drain(metrics_queue)[-1]

    # Assert: A clean terminal snapshot with both output channels present
    assert final.is_final
    assert final.error is None
    assert final.predictions is not None
    assert final.predictions.shape == (16, 2)
    assert math.isfinite(final.total_loss)


class _ConsumerDrainedRaceQueue(queue.Queue):  # type: ignore[type-arg]
    """Queue that reports Full once, then Empty on eviction, as if a consumer drained it."""

    def __init__(self) -> None:
        super().__init__(maxsize=200)
        self._reported_full = False
        self._reported_empty = False

    def put_nowait(self, item: TrainingMetrics) -> None:
        if not self._reported_full:
            self._reported_full = True
            raise queue.Full
        super().put_nowait(item)

    def get_nowait(self) -> TrainingMetrics:
        if not self._reported_empty:
            self._reported_empty = True
            raise queue.Empty
        return super().get_nowait()


def test_worker_push_survives_consumer_draining_queue_mid_overflow() -> None:
    """Confirms a Full-then-Empty race still delivers the snapshot instead of raising."""
    # Arrange
    racy_queue = _ConsumerDrainedRaceQueue()
    worker = _build_worker(total_epochs=1, metrics_queue=racy_queue)
    snapshot = TrainingMetrics(
        epoch=1, total_loss=0.5, bc_loss=0.2, physics_loss=0.3, elapsed_seconds=0.1
    )

    # Act
    worker._push_metrics(snapshot)

    # Assert
    assert racy_queue.get_nowait() is snapshot
