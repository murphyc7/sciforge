from abc import ABC, abstractmethod

import torch


class BasePhysicsSolver(ABC):
    """Abstract interface for scientific neural net partial differential equation constraints."""

    @abstractmethod
    def compute_residuals(
        self, x: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Calculates loss tensor mapping metrics for active physics environments.

        Args:
            x (torch.Tensor): Spatial coordinate tracking grid of shape `[Batch Size, 1]`.
            model (nn.Module): The neural network evaluating target fields at coordinates `x`.

        Returns:
            dict[str, torch.Tensor]: Map of string identifiers to calculated residual loss
                tensors (e.g., {"transport": tensor}). Each tensor matches the shape of `x`.
        """
        pass  # pragma: no cover


class ConstantFieldPhysics(BasePhysicsSolver):
    """Evaluates the 1D Drift-Diffusion equation under a static, uniform electric field.

    Enforces steady-state charge carrier conservation assuming a constant acceleration profile.
    The underlying governing partial differential equation is defined as:

    .. math::

       \mathcal{R}(x) = D_n \frac{d^2n}{dx^2} + \mu_n E \frac{dn}{dx} = 0

    Where :math:`\mu_n` is the mobility inversely proportional to the material's effective mass,
    :math:`D_n` is the diffusion coefficient linked via the Einstein relation, and :math:`E`
    is the static electric field.

    Attributes:
        q (float): Elementary electron charge constant in Coulombs (1.602e-19 C).
        kb_t (float): Thermal energy constant at 300K in Joules.
        mobility (float): Electron mobility in :math:`\text{m}^2/(\text{V}\cdot\text{s})`.
        diffusion_coeff (float): Diffusion coefficient in :math:`\text{m}^2/\text{s}`.
        electric_field (float): Applied background uniform electric field in V/m.
    """  # noqa: W605
    def __init__(
        self, effective_mass: float, permittivity: float, electric_field: float = 1.0e3
    ) -> None:
        """Initialises the constant-field drift-diffusion physics environment.

        Args:
            effective_mass (float): Material effective mass relative to electron mass (:math:`m_e`).
            permittivity (float): Relative static material macroscopic dielectric constant.
            electric_field (float, optional): Applied uniform electric field in V/m.
                Defaults to 1.0e3.
        """
        # Base physical constants
        self.q = 1.602e-19  # Electron charge (C)
        self.kb_t = 0.0259 * self.q  # Thermal energy at 300K (J)

        # Dynamic material property calculations derived from database metrics
        # Standard baseline silicon mobility scales inversely with effective mass ratio
        base_mobility = 0.14  # m^2/(V*s)
        self.mobility = base_mobility / (effective_mass + 1e-8)
        self.diffusion_coeff = self.mobility * (self.kb_t / self.q)

        # Electric field V/m
        self.electric_field = electric_field

    def compute_residuals(
        self, x: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Calculates the 1D transport residual error via automatic differentiation.

        Args:
            x (torch.Tensor): Spatial coordinate tracking grid of shape `[Batch Size, 1]`.
            model (nn.Module): Neural network tracking single-field carrier density :math:`n(x)`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing a single entry mapping the key
                "transport" to the calculated residual tensor of shape `[Batch Size, 1]`.
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
        return {"transport": residual}


class PoissonCoupledPhysics(BasePhysicsSolver):
    """Evaluates highly non-linear self-consistent coupled Poisson and Drift-Diffusion loops.

    Enforces space-charge electrostatic balance and current continuity simultaneously. The
    engine solves a coupled system of multi-field non-linear partial differential equations:

    .. math::

       \mathcal{R}_{\text{poisson}}(x) = \frac{d^2\phi}{dx^2} + \frac{q}{\epsilon_0 \epsilon_r}
       \left( N_D^+ - n(x) \right) = 0

    .. math::

       \mathcal{R}_{\text{transport}}(x) = D_n \frac{d^2n}{dx^2} + \mu_n \left(-\frac{d\phi}{dx}
       \right) \frac{dn}{dx} + \mu_n \left(-\frac{d^2\phi}{dx^2}\right)n(x) = 0

    Where :math:`\phi(x)` represents the local electrostatic potential, :math:`n(x)` is the
    carrier concentration, and :math:`N_D^+` is the background donor doping density.

    Attributes:
        q (float): Elementary electron charge constant in Coulombs (1.602e-19 C).
        kb_t (float): Thermal energy constant at 300K in Joules.
        eps0 (float): Vacuum permittivity constant in Farads per meter (8.854e-12 F/m).
        eps_r (float): Relative material static dielectric constant.
        nd (float): Uniform background n-type donor doping concentration in :math:`\text{m}^{-3}`.
        mobility (float): Electron mobility in :math:`\text{m}^2/(\text{V}\cdot\text{s})`.
        diffusion_coeff (float): Diffusion coefficient in :math:`\text{m}^2/\text{s}`.
    """  # noqa: W605

    def __init__(
        self, effective_mass: float, permittivity: float, donor_doping: float = 1.0e22
    ) -> None:
        """Initialises the self-consistent Poisson-coupled physics solver.

        Args:
            effective_mass (float): Material effective mass relative to electron mass (:math:`m_e`).
            permittivity (float): Relative static material macroscopic dielectric constant.
            donor_doping (float, optional): Background uniform ionized donor concentration in
                :math:`\text{m}^{-3}`. Defaults to 1.0e22.
        """
        self.q = 1.602e-19
        self.kb_t = 0.0259 * self.q
        self.eps0 = 8.854e-12
        self.eps_r = permittivity
        self.nd = donor_doping
        self.mobility = 0.14 / (effective_mass + 1e-8)
        self.diffusion_coeff = self.mobility * (self.kb_t / self.q)

    def compute_residuals(
        self, x: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Calculates independent residuals for both the Poisson and Transport loops.

        Extracts multi-variable outputs from the network nodes, evaluates higher-order spatial
        potentials, and tracks gradients using the autograd computational history graph.

        Args:
            x (torch.Tensor): Spatial coordinate tracking grid of shape `[Batch Size, 1]`.
            model (nn.Module): Neural network tracking multi-variable outputs where channel 0
                maps to carrier density :math:`n(x)` and channel 1 maps to potential :math:`\phi(x)`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing two keys:
                - "poisson": Electrostatic residual tensor of shape `[Batch Size, 1]`.
                - "transport": Coupled drift-diffusion conservation tensor of shape `[Batch Size, 1]`.
        """  # noqa: W605
        x.requires_grad_(True)
        predictions = model(x)

        n = predictions[:, 0:1]
        phi = predictions[:, 1:2]

        dn_dx = torch.autograd.grad(
            n, x, torch.ones_like(n), create_graph=True, retain_graph=True
        )[0]
        dphi_dx = torch.autograd.grad(
            phi, x, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]

        d2n_dx2 = torch.autograd.grad(
            dn_dx, x, torch.ones_like(dn_dx), create_graph=True, retain_graph=True
        )[0]
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx, x, torch.ones_like(dphi_dx), create_graph=True, retain_graph=True
        )[0]

        permittivity_factor = self.eps0 * self.eps_r
        poisson_residual = d2phi_dx2 + (self.q / permittivity_factor) * (self.nd - n)

        drift_term = self.mobility * (-dphi_dx * dn_dx - d2phi_dx2 * n)
        diffusion_term = self.diffusion_coeff * d2n_dx2
        transport_residual = diffusion_term + drift_term

        return {"poisson": poisson_residual, "transport": transport_residual}
