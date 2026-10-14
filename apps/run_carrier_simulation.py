#!/usr/bin/env python3
import argparse
import logging
import sys
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

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_criterion = torch.nn.MSELoss()

    logger.info("Starting Physics-Informed Neural Network optimisation loop...")
    start_time = time.time()

    # Training Loop Execution
    for epoch in range(args.epochs + 1):
        optimiser.zero_grad()

        loss_bc = mse_criterion(
            model(x_boundary_left), n_boundary_left
        ) + mse_criterion(model(x_boundary_right), n_boundary_right)

        residuals_dict = physics_engine.compute_residuals(x_interior, model)

        loss_physics = 0.0
        for key, res in residuals_dict.items():
            if key == "poisson":
                loss_physics += mse_criterion(res * 1.0e-3, torch.zeros_like(res))
            else:
                loss_physics += mse_criterion(res, torch.zeros_like(res))

        total_loss = loss_bc + loss_physics
        total_loss.backward()
        optimiser.step()

        if epoch % 100 == 0:
            logger.info(
                f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.4e} | BC Mismatch: {loss_bc.item():.4e}"
            )

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
    val_total = float(total_loss.item())
    val_bc = float(loss_bc.item())
    val_physics = (
        float(loss_physics.item())
        if isinstance(loss_physics, torch.Tensor)
        else float(loss_physics)
    )

    logger.info("Proceeding to MLOps database serialisation step...")
    tracker = SimulationTracker(db_client=db_client)
    tracker.register_run(
        material_id=args.material_id,
        engine_mode=args.engine,
        epochs=args.epochs,
        lr=1e-3,
        total_loss=val_total,
        bc_loss=val_bc,
        physics_loss=val_physics,
        start_time=start_time,
        model=model,
    )


if __name__ == "__main__":
    main()
