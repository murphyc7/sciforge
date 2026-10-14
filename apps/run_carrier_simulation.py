#!/usr/bin/env python3
import argparse
import logging
import queue
import sys
import threading
import time

import numpy as np
import torch

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    BasePhysicsSolver,
    BipolarCoupledPhysics,
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
from sciforge.matkit.carrier_pinn.training_worker import (
    PINNTrainingWorker,
    TrainingMetrics,
)
from sciforge.matkit.registry.manager import SimulationTracker
from sciforge.matkit.solvers.metrics import relative_l2_error
from sciforge.matkit.solvers.reference import (
    ReferenceSolution,
    solve_constant_field,
    solve_coupled_poisson,
)
from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel
from sciforge.visualisation.components import DualAxisTransportPlotter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description="SciForge PINN Solver Runner")
    parser.add_argument(
        "--material-id", type=str, default="mp-100", help="Target Material ID"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Optimisation steps")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["constant", "coupled", "bipolar"],
        default="constant",
        help="Physics mode selector",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the classical reference comparison (constant and coupled modes)",
    )
    parser.add_argument(
        "--poisson-weight",
        type=float,
        default=1.0e-3,
        help="Scale on the Poisson residual in the loss (squared in the loss term)",
    )
    args = parser.parse_args()

    db_client = DatabaseClient()
    with db_client.get_session() as session:
        material = (
            session.query(MaterialModel).filter_by(material_id=args.material_id).first()
        )
        if not material:
            logger.error(
                f"Material {args.material_id} missing. Run the materials pipeline first."
            )
            sys.exit(1)

        m_eff = material.effective_mass_me
        eps = material.permittivity
        formula = material.formula

    logger.info(
        f"Loaded constraints for {formula}: m*={m_eff}, eps={eps} | Mode: {args.engine.upper()}"
    )

    x_interior = torch.linspace(0.0, 1.0, 100, dtype=torch.float32).view(-1, 1)
    x_boundary_left = torch.tensor([[0.0]], dtype=torch.float32)
    x_boundary_right = torch.tensor([[1.0]], dtype=torch.float32)

    physics_engine: BasePhysicsSolver
    if args.engine == "constant":
        model = CarrierPINN(output_dim=1)
        physics_engine = ConstantFieldPhysics(effective_mass=m_eff, permittivity=eps)
        n_boundary_left = torch.tensor([[1.0]], dtype=torch.float32)
        n_boundary_right = torch.tensor([[0.0]], dtype=torch.float32)
    elif args.engine == "coupled":
        model = CarrierPINN(output_dim=2)
        physics_engine = PoissonCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        n_boundary_left = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        n_boundary_right = torch.tensor([[0.1, 0.5]], dtype=torch.float32)
    else:
        model = CarrierPINN(output_dim=3)
        physics_engine = BipolarCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        # Left boundary boundary conditions: [n_scaled, p_scaled, phi_scaled]
        n_boundary_left = torch.tensor([[1.0, 0.01, 0.0]], dtype=torch.float32)
        n_boundary_right = torch.tensor([[0.1, 1.0, 0.8]], dtype=torch.float32)

    logger.info("Starting Physics-Informed Neural Network optimisation loop...")

    # Run the optimisation on a background thread and stream its progress snapshots
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    worker = PINNTrainingWorker(
        model=model,
        physics_engine=physics_engine,
        x_interior=x_interior,
        x_left=x_boundary_left,
        x_right=x_boundary_right,
        n_left=n_boundary_left,
        n_right=n_boundary_right,
        total_epochs=args.epochs,
        metrics_queue=metrics_queue,
        stop_event=threading.Event(),
        lr=LEARNING_RATE,
        report_every=100,
        poisson_loss_weighting=args.poisson_loss_weighting,
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    start_time = time.time()
    thread.start()

    final: TrainingMetrics | None = None
    while final is None:
        try:
            snapshot = metrics_queue.get(timeout=1.0)
        except queue.Empty:
            if not thread.is_alive():
                break  # The worker exited without emitting a terminal snapshot
            continue
        if snapshot.is_final:
            final = snapshot
        else:
            logger.info(
                f"Epoch {snapshot.epoch:04d} | Total Loss: {snapshot.total_loss:.4e} "
                f"| BC Mismatch: {snapshot.bc_loss:.4e}"
            )
    thread.join()

    if final is None or final.error is not None:
        reason = final.error if final else "worker exited without a result"
        logger.error(f"Training failed: {reason}")
        sys.exit(1)

    # Sync eval, plotting, MLOPS registration phase
    logger.info("Simulation complete. Commencing evaluation and graphics generation...")

    # 1. Switch model to static evaluation mode and extract raw values safely
    model.eval()
    with torch.no_grad():
        final_predictions = model(x_interior)
        np_x = x_interior.cpu().numpy()
        np_out = final_predictions.cpu().numpy()

    # 1b. Validate against the classical reference solution (constant and coupled modes)
    reference: ReferenceSolution | None = None
    reference_overlay: dict[str, object] = {}
    if not args.skip_validation:
        grid = np_x.ravel().astype(np.float64)
        if isinstance(physics_engine, ConstantFieldPhysics):
            reference = solve_constant_field(
                physics_engine,
                n_left=float(n_boundary_left[0, 0]),
                n_right=float(n_boundary_right[0, 0]),
                x=grid,
            )
        elif isinstance(physics_engine, PoissonCoupledPhysics):
            reference = solve_coupled_poisson(
                physics_engine,
                n_left=float(n_boundary_left[0, 0]),
                n_right=float(n_boundary_right[0, 0]),
                phi_left=float(n_boundary_left[0, 1]),
                phi_right=float(n_boundary_right[0, 1]),
                x=grid,
            )

    if reference is not None:
        n_error = relative_l2_error(np_out[:, 0], reference.n)
        logger.info(f"Validation | relative L2 error vs reference (n): {n_error:.3e}")
        if reference.phi is not None:
            phi_error = relative_l2_error(np_out[:, 1], reference.phi)
            logger.info(
                f"Validation | relative L2 error vs reference (phi): {phi_error:.3e}"
            )
        # Reuse the plotter's overlay channel; subsample so markers stay legible, but
        # always keep the final point so a sharp boundary layer is not truncated.
        keep = np.union1d(np.arange(0, reference.x.size, 5), [reference.x.size - 1])
        reference_overlay = {
            "x_exp": reference.x[keep],
            "n_exp": reference.n[keep],
            "exp_label": "Classical reference",
        }

    # 2. Initialise scientific plotting layer
    plotter = DualAxisTransportPlotter()

    if args.engine == "constant":
        plotter.render(
            save_path="example_visualisations/simulation_constant.png",
            x=np_x,
            n=np_out[:, 0:1],
            title=f"Steady-State 1D Transport Profiles ({formula})",
            **reference_overlay,
        )
        logger.info("Constant field visualisation saved successfully.")
    elif args.engine == "coupled":
        plotter.render(
            save_path="example_visualisations/simulation_coupled.png",
            x=np_x,
            n=np_out[:, 0:1],
            phi=np_out[:, 1:2],
            title=f"Self-Consistent Multi-Field Solutions ({formula})",
            **reference_overlay,
        )
        logger.info("Coupled self-consistent field visualisation saved successfully.")
    else:
        # Column 0 = electrons n(x) | Column 1 = holes p(x) | Column 2 = potential phi(x)
        plotter.render(
            save_path="example_visualisations/simulation_bipolar.png",
            x=np_x,
            n=np_out[:, 0:1],  # Electron tracking matrix array slice
            p=np_out[:, 1:2],  # Hole tracking matrix array slice
            phi=np_out[:, 2:3],  # Electrostatic potential matrix array slice
            title=f"Bipolar Self-Consistent Transport Profiles ({formula})",
        )
        logger.info(
            "Bipolar multi-field self-consistent visualisation saved successfully."
        )

    # 3. Package numeric primitives for MLOps ledger tracking
    val_total = final.total_loss
    val_bc = final.bc_loss
    val_physics = final.physics_loss

    logger.info("Proceeding to MLOps database serialisation step...")
    tracker = SimulationTracker(db_client=db_client)
    tracker.register_run(
        material_id=args.material_id,
        engine_mode=args.engine,
        epochs=args.epochs,
        lr=LEARNING_RATE,
        total_loss=val_total,
        bc_loss=val_bc,
        physics_loss=val_physics,
        start_time=start_time,
        model=model,
    )


if __name__ == "__main__":
    main()
