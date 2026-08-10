"""MLOps infrastructure layer for model artifact serialisation and metadata ledger tracking.

Provides automated interfaces to capture running logs, serialise deep learning states
into industry-standard ONNX profiles, and commit metrics straight to SQL stores.
"""

import logging
import os
import time

import torch

from sciforge.matkit.utils.db_client import DatabaseClient, SimulationRegistryModel

logger = logging.getLogger(__name__)


class SimulationTracker:
    """Manages the lifecycle serialisation and database registration for tracking runs."""

    def __init__(
        self, db_client: DatabaseClient, artifact_dir: str = "artifacts/models"
    ) -> None:
        """Initialises the MLOps tracking interface components.

        Args:
            db_client (DatabaseClient): Active relational data infrastructure manager.
            artifact_dir (str, optional): Target local directory mapping model binaries.
                Defaults to 'artifacts/models'.
        """
        self.db_client = db_client
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def _export_to_onnx(self, model: torch.nn.Module, run_uuid: str) -> str:
        """Serialises the dynamic PyTorch model into a portable static ONNX file template.

        Args:
            model (torch.nn.Module): The optimised PyTorch network weights instance.
            run_uuid (str): Unique timestamp string used to partition the output file name.

        Returns:
            str: Relative system filename target destination path.
        """
        artifact_name = f"pinn_model_{run_uuid}.onnx"
        full_path = os.path.join(self.artifact_dir, artifact_name)

        model.eval()
        dummy_input = torch.linspace(0.0, 1.0, 10, dtype=torch.float32).view(-1, 1)

        logger.info(
            f"Executing static path ONNX graph compilation to target: {full_path}"
        )
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                full_path,
                export_params=True,
                opset_version=15,
                input_names=["spatial_x"],
                output_names=["predicted_fields"],
                dynamic_axes={"spatial_x": {0: "batch_size"}},
            )

        return full_path

    def register_run(
        self,
        material_id: str,
        engine_mode: str,
        epochs: int,
        lr: float,
        total_loss: float,
        bc_loss: float,
        physics_loss: float,
        start_time: float,
        model: torch.nn.Module,
    ) -> int:
        """Saves model parameters to disk and logs operational statistics to the SQL database.

        Args:
            material_id (str): Reference database identifier tracking target material metrics.
            engine_mode (str): Selected solver profile flag chosen ('constant'/'coupled').
            epochs (int): Processing loop step targets completed.
            lr (float): Applied optimisation boundary adjustments.
            total_loss (float): Aggregate minimisation error profile captured.
            bc_loss (float): Target boundary data constraints mismatch error scores.
            physics_loss (float): Calculated differential equation residual values.
            start_time (float): Precision epoch clock reference timestamp.
            model (torch.nn.Module): The active PyTorch neural network solver weights.

        Returns:
            int: Automated sequential run_id primary key index tracking the logged row.
        """
        duration = time.time() - start_time
        run_timestamp = str(int(time.time()))

        # Export exactly ONE clean file per engine run
        artifact_path = self._export_to_onnx(model, run_timestamp)

        record = SimulationRegistryModel(
            material_id=material_id,
            engine_mode=engine_mode,
            epochs_trained=epochs,
            learning_rate=lr,
            final_total_loss=total_loss,
            final_bc_loss=bc_loss,
            final_physics_loss=physics_loss,
            execution_time_seconds=duration,
            model_artifact_path=artifact_path,
        )

        logger.info(
            "Injecting operational metadata metrics into the relational store..."
        )

        # Hard imperative session management matching your working ETL setup
        session = self.db_client.get_session()
        try:
            session.add(record)
            session.commit()  # Forces the database socket to flush natively to PostgreSQL disk
            generated_id = record.run_id
        except Exception as e:
            session.rollback()
            logger.error(f"Relational transaction aborted and rolled back: {e}")
            raise e
        finally:
            session.close()

        logger.info(f"Run successfully registered under reference ID: {generated_id}")
        return generated_id
