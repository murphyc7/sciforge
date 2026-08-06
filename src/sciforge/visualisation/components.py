"""Concrete scientific plot components designed for data model visualisation.

Provides multi-field dual-axis line drawing systems optimised to overlay
electrostatic configurations and transport density profiles safely.
"""

import matplotlib.pyplot as plt
import numpy as np

from sciforge.visualisation.base_plotter import BasePlotter


class DualAxisTransportPlotter(BasePlotter):
    """Renders highly legible 1D spatial transport curves across dual axes.

    Can scale dynamically to display either independent transport lines or coupled
    potential/density fields side-by-side on the same coordinate grid.
    """

    def render(self, save_path: str, **kwargs: np.ndarray) -> None:
        """Generates the figure and exports a high-density vector graphic file.

        Args:
            save_path (str): Complete file name destination path.
            **kwargs: Expected numpy arrays containing:
                - 'x': Spatial coordinates vector grid.
                - 'n': Calculated carrier density dataset matrix.
                - 'phi' (optional): Local electrostatic potential array.

        Raises:
            KeyError: If mandatory parameters 'x' or 'n' are missing from execution calls.
        """
        if "x" not in kwargs or "n" not in kwargs:
            raise KeyError(
                "Plotting routine requires explicit 'x' and 'n' array matrix profiles."
            )

        x = kwargs["x"].flatten()
        n = kwargs["n"].flatten()
        phi = kwargs.get("phi")

        # Create an isolated figure context to prevent cross-test memory contamination
        fig, ax1 = plt.subplots(figsize=self.fig_size)

        # Primary axis: Carrier Concentration
        color_n = "#1f77b4"  # High-contrast scientific blue
        line1 = ax1.plot(
            x,
            n,
            color=color_n,
            linestyle="-",
            linewidth=1.5,
            label="Carrier Density ($n$)",
        )
        ax1.set_xlabel("Normalised Position ($x / L$)")
        ax1.set_ylabel("Normalised Density ($n / N_0$)", color=color_n)
        ax1.tick_params(axis="y", labelcolor=color_n)

        lines = line1

        # Secondary axis: Electrostatic Potential (If Coupled Mode Is Active)
        if phi is not None:
            phi = phi.flatten()
            ax2 = ax1.twinx()  # Generate aligned secondary y-axis overlay
            color_phi = "#d62728"  # High-contrast scientific red
            line2 = ax2.plot(
                x,
                phi,
                color=color_phi,
                linestyle="--",
                linewidth=1.5,
                label="Potential ($\phi$)",  # noqa: W605
            )
            ax2.set_ylabel("Electrostatic Potential ($\phi$ (V))", color=color_phi)  # noqa: W605
            ax2.tick_params(axis="y", labelcolor=color_phi)
            ax2.grid(False)  # Turn off secondary grid lines to avoid visual clutter
            lines = line1 + line2

        # Unified legend allocation
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        # Export graphics layout cleanly
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
