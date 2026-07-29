#!/usr/bin/env python3
import argparse
import logging
import sys

import torch
import torch.nn as nn

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import DriftDiffusionPhysics
from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SciForge PINN Solver Runner")
    parser.add_argument(
        "--material-id", type=str, default="mp-100", help="Target Material ID"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Optimization steps")
    args = parser.parse_args()

    # Step 1: Query material metrics from PostgreSQL
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

    logger.info(f"Loaded constraints for {formula}: m*={m_eff}, eps={eps}")

    # Step 2: Initialize PyTorch SciML Models
    model = CarrierPINN()
    physics_engine = DriftDiffusionPhysics(effective_mass=m_eff, permittivity=eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()

    # Step 3: Define spatial boundary grid (x scaled from 0.0 to 1.0 micron)
    x_interior = torch.linspace(0.0, 1.0, 100, dtype=torch.float32).view(-1, 1)

    # Set hard Dirichlet Boundary Conditions: n(0) = 1.0, n(1) = 0.0
    x_boundary_left = torch.tensor([[0.0]], dtype=torch.float32)
    n_boundary_left = torch.tensor([[1.0]], dtype=torch.float32)

    x_boundary_right = torch.tensor([[1.0]], dtype=torch.float32)
    n_boundary_right = torch.tensor([[0.0]], dtype=torch.float32)

    logger.info("Starting Physics-Informed Neural Network optimization loop...")

    # Step 4: Training Loop
    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()

        # Evaluate structural boundary mismatch (Data Loss)
        pred_left = model(x_boundary_left)
        pred_right = model(x_boundary_right)
        loss_boundary = mse_criterion(pred_left, n_boundary_left) + mse_criterion(
            pred_right, n_boundary_right
        )

        # Evaluate internal PDE constraint mismatch (Physics Loss)
        pde_residual = physics_engine.compute_pde_residual(x_interior, model)
        loss_physics = mse_criterion(pde_residual, torch.zeros_like(pde_residual))

        # Combined loss structure
        total_loss = loss_boundary + loss_physics
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            logger.info(
                f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | "
                f"Boundary Loss: {loss_boundary.item():.6f} | Physics Loss: {loss_physics.item():.6f}"
            )


if __name__ == "__main__":
    main()
