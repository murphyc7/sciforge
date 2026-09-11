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
    r"""Evaluates the 1D Drift-Diffusion equation under a static, uniform electric field.

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
    """

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
        # Scale parameters
        self.L_scale = 1.0e-6  # 1 um scaling factor

        # Base physical constants
        self.q = 1.602e-19  # Electron charge (C)
        self.kb_t = 0.0259 * self.q  # Thermal energy at 300K (J)

        # Dynamic material property calculations derived from database metrics
        # Standard baseline silicon mobility scales inversely with effective mass ratio
        base_mobility = 0.14  # m^2/(V*s)
        self.mobility = base_mobility / (effective_mass + 1e-8)
        self.diffusion_coeff = self.mobility * (self.kb_t / self.q)
        self.electric_field = electric_field

        # Calculate the exact mathematical magnitude of the second derivative when evaluated
        # on normalised 0.0 to 1.0 grid space
        coeff_d2n_scaled = self.diffusion_coeff / (self.L_scale**2)

        # Create an automatic structural multiplier to bound your physics loss near ~1.0
        self.loss_normaliser = 1.0 / coeff_d2n_scaled

    def compute_residuals(
        self, x_scaled: torch.Tensor, model: torch.nn.Module
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
        x_scaled.requires_grad_(True)

        # Evaluate model prediction: n = f(x)
        n = model(x_scaled)

        # Compute first derivative: dn/dx
        dn_dx = torch.autograd.grad(
            outputs=n,
            inputs=x_scaled,
            grad_outputs=torch.ones_like(n),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute second derivative: d2n/dx2
        d2n_dx2 = torch.autograd.grad(
            outputs=dn_dx,
            inputs=x_scaled,
            grad_outputs=torch.ones_like(dn_dx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Convert derivatives back to physical space based on length scale rules
        dn_dx_phys = dn_dx / self.L_scale
        d2n_dx2_phys = d2n_dx2 / (self.L_scale**2)

        # Assume net generation-recombination rate R(x) is negligible for steady-state demonstration
        # PDE Residual: D_n * (d2n/dx2) + mu_n * E * (dn/dx) = 0
        drift_term = self.mobility * self.electric_field * dn_dx_phys
        diffusion_term = self.diffusion_coeff * d2n_dx2_phys

        # Multiply the entire residual by normaliser
        # This scales the second derivative down, removing the flat-line penalty
        residual = (diffusion_term + drift_term) * self.loss_normaliser
        return {"transport": residual}


class PoissonCoupledPhysics(BasePhysicsSolver):
    r"""Evaluates highly non-linear self-consistent coupled Poisson and Drift-Diffusion loops.

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
    """

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
        # Scaling parameters
        self.L_scale = 1.0e-6  # 1 micron scaling factor
        self.N_scale = 1.0e22  # 1e22 m^-3 carrier scaling factor

        # Physical parameters
        self.q = 1.602e-19
        self.kb_t = 0.0259 * self.q
        self.eps0 = 8.854e-12
        self.eps_r = permittivity
        self.nd_scaled = (
            donor_doping / self.N_scale
        )  # Background uniform doping scaled to 1.0
        self.mobility = 0.14 / (effective_mass + 1e-8)
        self.diffusion_coeff = self.mobility * (self.kb_t / self.q)

    def compute_residuals(
        self, x_scaled: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        r"""Calculates independent residuals for both the Poisson and Transport loops.

        Extracts multi-variable outputs from the network nodes, evaluates higher-order spatial
        potentials, and tracks gradients using the autograd computational history graph.

        Args:
            x_scaled (torch.Tensor): Scaled spatial coordinate tracking grid of shape `[Batch Size, 1]`.
            model (nn.Module): Neural network tracking multi-variable outputs where channel 0
                maps to carrier density :math:`n(x)` and channel 1 maps to potential :math:`\phi(x)`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing two keys:
                - "poisson": Electrostatic residual tensor of shape `[Batch Size, 1]`.
                - "transport": Coupled drift-diffusion conservation tensor of shape `[Batch Size, 1]`.
        """
        x_scaled.requires_grad_(True)
        predictions = model(x_scaled)

        n_scaled = predictions[:, 0:1]
        phi = predictions[:, 1:2]

        dn_dx = torch.autograd.grad(
            n_scaled,
            x_scaled,
            torch.ones_like(n_scaled),
            create_graph=True,
            retain_graph=True,
        )[0]
        dphi_dx = torch.autograd.grad(
            phi, x_scaled, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]

        d2n_dx2 = torch.autograd.grad(
            dn_dx,
            x_scaled,
            torch.ones_like(dn_dx),
            create_graph=True,
            retain_graph=True,
        )[0]
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx,
            x_scaled,
            torch.ones_like(dphi_dx),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Scaled Equation 1: Poisson Residual
        # d2phi/dx^2 + L^2 * (q * N_scale / eps) * (Nd - n) = 0
        poisson_constant = (
            (self.L_scale**2) * (self.q * self.N_scale) / (self.eps0 * self.eps_r)
        )
        poisson_residual = d2phi_dx2 + poisson_constant * (self.nd_scaled - n_scaled)

        # Scaled Equation 2: Transport Residual
        # Dn * d2n/dx^2 - mu * dphi/dx * dn/dx - mu * d2phi/dx^2 * n = 0
        # Multiplied by L_scale^2 to match the matrix order of dimensions
        drift_term = self.mobility * (-dphi_dx * dn_dx - d2phi_dx2 * n_scaled)
        diffusion_term = self.diffusion_coeff * d2n_dx2
        transport_residual = diffusion_term + drift_term

        return {"poisson": poisson_residual, "transport": transport_residual}


class BipolarCoupledPhysics(BasePhysicsSolver):
    r"""Numerically stable, self-consistent coupled Poisson, Electron, and Hole Continuity solver.

    Enforces total space-charge electrostatic balance and non-equilibrium current
    conservation simultaneously. This strategy uses automated differentiation graphs
    to evaluate three tightly coupled non-linear partial differential equations:

    .. math::

       \mathcal{R}_{\text{poisson}}(x) = \frac{d^2\phi}{dx^2} + \frac{q}{\epsilon_0 \epsilon_r}
       \left( N_D^+ - N_A^- + p(x) - n(x) \right) = 0

    .. math::

       \mathcal{R}_{\text{electron}}(x) = D_n \frac{d^2n}{dx^2} - \mu_n \frac{d\phi}{dx}\frac{dn}{dx}
       - \mu_n \frac{d^2\phi}{dx^2}n(x) - U_{\text{SRH}}(x) = 0

    .. math::

       \mathcal{R}_{\text{hole}}(x) = D_p \frac{d^2p}{dx^2} + \mu_p \frac{d\phi}{dx}\frac{dp}{dx}
       + \mu_p \frac{d^2\phi}{dx^2}p(x) - U_{\text{SRH}}(x) = 0

    Where generation-recombination currents are driven via a non-linear Shockley-Read-Hall
    (SRH) trap-assisted recombination profile:

    .. math::

       U_{\text{SRH}}(x) = \frac{n(x)p(x) - n_i^2}{\tau_p (n(x) + n_i) + \tau_n (p(x) + n_i)}
    """

    def __init__(
        self, effective_mass: float, permittivity: float, donor_doping: float = 1.0e22
    ) -> None:
        """Initialises the self-consistent bipolar Poisson-recombination physics engine.

        Args:
            effective_mass (float): Material effective mass relative to electron mass (:math:`m_e`).
            permittivity (float): Relative static material macroscopic dielectric constant.
            donor_doping (float, optional): Background uniform ionized donor concentration in
                :math:`\text{m}^{-3}`. Defaults to 1.0e22.
        """
        self.L_scale = 1.0e-6
        self.N_scale = 1.0e22
        self.q = 1.602e-19
        self.kb_t = 0.0259 * self.q
        self.eps0 = 8.854e-12
        self.eps_r = permittivity

        # Mobility and Diffusion parameters for both carriers
        self.mu_n = 0.14 / (effective_mass + 1e-8)
        self.mu_p = (
            self.mu_n / 3.0
        )  # Physical hole mobility is roughly 1/3 of electron mobility
        self.dn = self.mu_n * (self.kb_t / self.q)
        self.dp = self.mu_p * (self.kb_t / self.q)

        # Recombination thresholds
        self.tau_n = 1.0e-6  # Electron lifetime (s)
        self.tau_p = 1.0e-6  # Hole lifetime (s)
        self.ni = 1.5e16  # Intrinsic carrier density (m^-3)
        self.nd_scaled = donor_doping / self.N_scale

    def compute_residuals(
        self, x_scaled: torch.Tensor, model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Calculates independent residuals for Poisson, Electron, and Hole current loops.

        Extracts three distinct multi-variable channels from the target network node outputs,
        maps physical spatial coordinates using the chain rule, and aggregates
        non-linear current recombination losses safely into the computational graph.

        Args:
            x_scaled (torch.Tensor): Bounded spatial coordinate tracking grid of shape `[Batch Size, 1]`.
            model (nn.Module): Neural network tracking multi-variable outputs where:
                - Channel 0 maps to electron concentration: :math:`n(x)`.
                - Channel 1 maps to hole concentration: :math:`p(x)`.
                - Channel 2 maps to local electrostatic potential: :math:`\phi(x)`.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing three physical validation keys:
                - "poisson": Bipolar space-charge electrostatic residual matrix of shape `[Batch Size, 1]`.
                - "electron": Electron continuity conservation residual matrix of shape `[Batch Size, 1]`.
                - "hole": Hole continuity conservation residual matrix of shape `[Batch Size, 1]`.
        """
        x_scaled.requires_grad_(True)
        predictions = model(x_scaled)

        # Slice output matrix nodes to map three distinct fields simultaneously
        n_scaled = predictions[:, 0:1]
        p_scaled = predictions[:, 1:2]
        phi = predictions[:, 2:3]

        # First derivatives
        dn_dx = torch.autograd.grad(
            n_scaled,
            x_scaled,
            torch.ones_like(n_scaled),
            create_graph=True,
            retain_graph=True,
        )
        dp_dx = torch.autograd.grad(
            p_scaled,
            x_scaled,
            torch.ones_like(p_scaled),
            create_graph=True,
            retain_graph=True,
        )
        dphi_dx = torch.autograd.grad(
            phi, x_scaled, torch.ones_like(phi), create_graph=True, retain_graph=True
        )

        # Second derivatives
        d2n_dx2 = torch.autograd.grad(
            dn_dx,
            x_scaled,
            torch.ones_like(dn_dx),
            create_graph=True,
            retain_graph=True,
        )
        d2dp_dx2 = torch.autograd.grad(
            dp_dx,
            x_scaled,
            torch.ones_like(dp_dx),
            create_graph=True,
            retain_graph=True,
        )
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx,
            x_scaled,
            torch.ones_like(dphi_dx),
            create_graph=True,
            retain_graph=True,
        )

        # Restore physical dimensions for calculus evaluation loops
        dn_dx_phys = dn_dx / self.L_scale
        dp_dx_phys = dp_dx / self.L_scale
        dphi_dx_phys = dphi_dx / self.L_scale
        d2n_dx2_phys = d2n_dx2 / (self.L_scale**2)
        d2dp_dx2_phys = d2dp_dx2 / (self.L_scale**2)
        d2phi_dx2_phys = d2phi_dx2 / (self.L_scale**2)

        n_phys = n_scaled * self.N_scale
        p_phys = p_scaled * self.N_scale

        # Non-linear SRH Recombination Calculation
        u_srh = (n_phys * p_phys - self.ni**2) / (
            self.tau_p * (n_phys + self.ni) + self.tau_n * (p_phys + self.ni)
        )

        # 1. Bipolar Poisson Residual
        poisson_constant = self.q / (self.eps0 * self.eps_r)
        poisson_residual = (
            d2phi_dx2_phys
            + poisson_constant * (self.nd_scaled * self.N_scale - n_phys + p_phys)
        ) * 1.0e-3

        # 2. Electron Continuity Residual
        electron_residual = (
            self.dn * d2n_dx2_phys
            - self.mu_n * (dphi_dx_phys * dn_dx_phys + d2phi_dx2_phys * n_phys)
            - u_srh
        )

        # 3. Hole Continuity Residual
        hole_residual = (
            self.dp * d2dp_dx2_phys
            + self.mu_p * (dphi_dx_phys * dp_dx_phys + d2phi_dx2_phys * p_phys)
            - u_srh
        )

        return {
            "poisson": poisson_residual,
            "electron": electron_residual,
            "hole": hole_residual,
        }
