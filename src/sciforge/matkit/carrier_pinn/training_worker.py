"""Decoupled, thread-safe execution harness for Physics-Informed Neural Network training.

Separates the PyTorch optimisation loop from any calling interface (CLI, web dashboard,
notebook) by streaming progress snapshots through a thread-safe queue. This lets a
long-running training job execute on a background thread while the caller — e.g. a Dash
callback — remains free to serve the page and poll for progress.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from sciforge.matkit.carrier_pinn.physics import BasePhysicsSolver

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """A single progress snapshot emitted by a running PINN optimisation loop.

    Attributes:
        epoch (int): Completed optimisation step index.
        total_loss (float): Aggregate minimisation loss at this epoch.
        bc_loss (float): Boundary condition mismatch component of the loss.
        physics_loss (float): Physical PDE residual component of the loss.
        elapsed_seconds (float): Wall-clock time since the loop started.
        is_final (bool): True once the loop has stopped (completed, cancelled, or
            failed) and no further metrics will be queued. Defaults to False.
        predictions (np.ndarray | None): Evaluation-mode model predictions over the
            interior grid, populated only on the terminal snapshot. Defaults to None.
        error (str | None): Populated with an exception message if the loop aborted
            early. Defaults to None.
    """

    epoch: int
    total_loss: float
    bc_loss: float
    physics_loss: float
    elapsed_seconds: float
    is_final: bool = False
    predictions: np.ndarray | None = None
    error: str | None = None


class PINNTrainingWorker:
    """Runs a Carrier PINN optimisation loop on a background thread.

    Decouples the PyTorch training loop from the calling interface: `run` is intended to
    execute as a `threading.Thread` target. Progress is communicated back to the caller
    exclusively through `metrics_queue`, so the calling (UI) thread never touches the
    model weights or optimiser state while training is in flight — it only reads the
    immutable `TrainingMetrics` snapshots pulled off the queue.
    """

    def __init__(
        self,
        model: nn.Module,
        physics_engine: BasePhysicsSolver,
        x_interior: torch.Tensor,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        n_left: torch.Tensor,
        n_right: torch.Tensor,
        total_epochs: int,
        metrics_queue: "queue.Queue[TrainingMetrics]",
        stop_event: threading.Event,
        lr: float = 1e-3,
        report_every: int = 5,
    ) -> None:
        """Initialises the decoupled training worker.

        Args:
            model (nn.Module): The CarrierPINN network instance to optimise in-place.
            physics_engine (BasePhysicsSolver): The selected physics residual backend.
            x_interior (torch.Tensor): Interior collocation grid, shape [N, 1].
            x_left (torch.Tensor): Left boundary coordinate tensor.
            x_right (torch.Tensor): Right boundary coordinate tensor.
            n_left (torch.Tensor): Left boundary target values.
            n_right (torch.Tensor): Right boundary target values.
            total_epochs (int): Total optimisation steps to execute.
            metrics_queue (queue.Queue[TrainingMetrics]): Thread-safe sink that progress
                snapshots are pushed onto in real time.
            stop_event (threading.Event): Cooperative cancellation flag checked between
                epochs so a caller can request early termination.
            lr (float, optional): Adam optimiser learning rate. Defaults to 1e-3.
            report_every (int, optional): Emit a metrics snapshot every N epochs, in
                addition to the first and final epoch. Defaults to 5.
        """
        self.model = model
        self.physics_engine = physics_engine
        self.x_interior = x_interior
        self.x_left = x_left
        self.x_right = x_right
        self.n_left = n_left
        self.n_right = n_right
        self.total_epochs = total_epochs
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.lr = lr
        self.report_every = max(1, report_every)

    def _push_metrics(self, metrics: TrainingMetrics) -> None:
        """Pushes a metrics snapshot onto the queue, dropping the oldest entry if full.

        A bounded queue combined with a drop-oldest policy guarantees the frontend always
        sees the most recent progress, even if the polling thread temporarily falls behind
        on consumption instead of blocking the training loop.

        Args:
            metrics (TrainingMetrics): The snapshot to enqueue.
        """
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            try:
                self.metrics_queue.get_nowait()
            except queue.Empty:
                pass
            self.metrics_queue.put_nowait(metrics)

    def run(self) -> None:
        """Executes the PINN optimisation loop on the calling thread.

        Intended to be used as a `threading.Thread(target=worker.run)` entry point. Streams
        a `TrainingMetrics` snapshot onto `metrics_queue` every `report_every` epochs, and
        always emits exactly one terminal snapshot (`is_final=True`) carrying
        evaluation-mode predictions over the interior grid — whether the loop completed
        normally, was cancelled via `stop_event`, or raised an exception.
        """
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_criterion = nn.MSELoss()
        start_time = time.time()

        logger.info(
            f"Background training thread starting {self.total_epochs}-epoch "
            "optimisation loop."
        )

        epoch = 0
        total_loss = torch.tensor(0.0)
        loss_bc = torch.tensor(0.0)
        loss_physics: torch.Tensor | float = 0.0

        try:
            for epoch in range(self.total_epochs + 1):
                if self.stop_event.is_set():
                    logger.info(f"Training cancelled by caller at epoch {epoch}.")
                    break

                optimiser.zero_grad()
                loss_bc = mse_criterion(
                    self.model(self.x_left), self.n_left
                ) + mse_criterion(self.model(self.x_right), self.n_right)

                residuals_dict = self.physics_engine.compute_residuals(
                    self.x_interior, self.model
                )

                loss_physics = 0.0
                for key, res in residuals_dict.items():
                    if key == "poisson":
                        loss_physics += mse_criterion(
                            res * 1.0e-3, torch.zeros_like(res)
                        )
                    else:
                        loss_physics += mse_criterion(res, torch.zeros_like(res))

                total_loss = loss_bc + loss_physics
                total_loss.backward()
                optimiser.step()

                if epoch == 0 or epoch % self.report_every == 0:
                    self._push_metrics(
                        TrainingMetrics(
                            epoch=epoch,
                            total_loss=float(total_loss.item()),
                            bc_loss=float(loss_bc.item()),
                            physics_loss=float(
                                loss_physics.item()
                                if isinstance(loss_physics, torch.Tensor)
                                else loss_physics
                            ),
                            elapsed_seconds=time.time() - start_time,
                        )
                    )

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(self.x_interior).cpu().numpy()

            self._push_metrics(
                TrainingMetrics(
                    epoch=epoch,
                    total_loss=float(total_loss.item()),
                    bc_loss=float(loss_bc.item()),
                    physics_loss=float(
                        loss_physics.item()
                        if isinstance(loss_physics, torch.Tensor)
                        else loss_physics
                    ),
                    elapsed_seconds=time.time() - start_time,
                    is_final=True,
                    predictions=predictions,
                )
            )
            logger.info(f"Training thread completed at epoch {epoch}.")

        except Exception as e:
            logger.error(f"Training thread aborted with an exception: {e}")
            self._push_metrics(
                TrainingMetrics(
                    epoch=epoch,
                    total_loss=float("nan"),
                    bc_loss=float("nan"),
                    physics_loss=float("nan"),
                    elapsed_seconds=time.time() - start_time,
                    is_final=True,
                    error=str(e),
                )
            )
