import torch


class DriftDiffusionPhysics:
    """Evaluates the physical partial differential equation residuals for carrier transport.

    This class links semiconductor device constraints (effective mass, permittivity)
    to a PyTorch optimisation graph to solve steady-state drift-diffusion equations.

    Args:
        effective_mass (float): The material effective mass relative to electron mass (m_e).
        permittivity (float): The relative material permittivity (dielectric constant).
        electric_field (float, optional): Applied electric field in V/m. Defaults to 1.0e3.
    """
    def __init__(
        self, effective_mass: float, permittivity: float, electric_field: float = 1.0e3
    ) -> None:
        # Base physical constants
        self.q = 1.602e-19  # Electron charge (C)
        self.kb_t = 0.0259 * self.q  # Thermal energy at 300K (J)

        # Dynamic material property calculations derived from database metrics
        # Standard baseline silicon mobility scales inversely with effective mass ratio
        base_mobility = 0.14  # m^2/(V*s)
        self.mobility = base_mobility / (effective_mass + 1e-8)
        self.diffusion_coeff = self.mobility * (self.kb_t / self.q)

        self.electric_field = electric_field  # V/m

    def compute_pde_residual(
        self, x: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """Calculates the physical residual error of the drift-diffusion PDE using autograd.

        Computes the second-order spatial derivative to enforce:
        D_n * (d2n/dx2) + mu_n * E * (dn/dx) = 0

        Args:
            x (torch.Tensor): Spatial coordinate inputs of shape [Batch, 1].
            model (torch.nn.Module): The neural network predicting carrier density n(x).

        Returns:
            torch.Tensor: The physical residual tensor evaluated at each input point.
        """
        # Force tracker graph inclusion for spatial coordinates
        x.requires_grad_(True)

        # Evaluate model prediction: n = f(x)
        n = model(x)

        # Compute first derivative: dn/dx
        dn_dx = torch.autograd.grad(
            outputs=n,
            inputs=x,
            grad_outputs=torch.ones_like(n),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute second derivative: d2n/dx2
        d2n_dx2 = torch.autograd.grad(
            outputs=dn_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dn_dx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Assume net generation-recombination rate R(x) is negligible for steady-state demonstration
        # PDE Residual: D_n * (d2n/dx2) + mu_n * E * (dn/dx) = 0
        drift_term = self.mobility * self.electric_field * dn_dx
        diffusion_term = self.diffusion_coeff * d2n_dx2

        residual = diffusion_term + drift_term
        return residual
