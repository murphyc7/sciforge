#!/usr/bin/env python3
import argparse
import logging
import sys

import torch
import torch.nn as nn

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
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
    parser.add_argument("--epochs", type=int, default=500, help="Optimisation steps")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["constant", "coupled"],
        default="constant",
        help="Physics mode selector",
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
        f"Loaded constraints for {formula}: m*={m_eff}, eps={eps} | Engine: {args.engine.upper()}"
    )

    # Both models now receive a stable spatial coordinate grid scaled between 0.0 and 1.0
    x_interior = torch.linspace(0.0, 1.0, 100, dtype=torch.float32).view(-1, 1)
    x_boundary_left = torch.tensor([[0.0]], dtype=torch.float32)
    x_boundary_right = torch.tensor([[1.0]], dtype=torch.float32)

    if args.engine == "constant":
        model = CarrierPINN(output_dim=1)
        physics_engine = ConstantFieldPhysics(effective_mass=m_eff, permittivity=eps)
        n_boundary_left = torch.tensor([[1.0]], dtype=torch.float32)
        n_boundary_right = torch.tensor([[0.0]], dtype=torch.float32)
    else:
        model = CarrierPINN(output_dim=2)
        physics_engine = PoissonCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        # Normalized boundaries matching our 1.0 scaling factor rules
        n_boundary_left = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32
        )  # n=1.0, phi=0.0V
        n_boundary_right = torch.tensor(
            [[0.1, 0.5]], dtype=torch.float32
        )  # n=0.1, phi=0.5V

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()

    logger.info("Starting Physics-Informed Neural Network optimisation loop...")

    for epoch in range(args.epochs + 1):
        optimiser.zero_grad()

        # 1. Evaluate boundary mismatch
        loss_bc = mse_criterion(
            model(x_boundary_left), n_boundary_left
        ) + mse_criterion(model(x_boundary_right), n_boundary_right)

        # 2. Compute stable scaled residuals
        residuals_dict = physics_engine.compute_residuals(x_interior, model)

        loss_physics = 0.0
        for key, res in residuals_dict.items():
            if key == "poisson":
                # Scale Poisson loss slightly to balance with the transport gradients
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


if __name__ == "__main__":
    main()
