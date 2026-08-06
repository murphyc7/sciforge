"""Concrete scientific plot components designed for data model visualisation."""

from typing import Any

import matplotlib.pyplot as plt

from sciforge.visualisation.base_plotter import BasePlotter


class DualAxisTransportPlotter(BasePlotter):
    """Renders highly legible 1D spatial transport curves across dual axes.

    Supports dynamic title configuration and structural experimental data overlays
    with custom labeling injection.
    """

    def render(self, save_path: str, **kwargs: Any) -> None:
        """Generates the figure and exports a high-density vector graphic file.

        Args:
            save_path (str): Complete file name destination path.
            **kwargs: Expected data structures containing:
                - 'x' (np.ndarray): Spatial coordinates vector grid.
                - 'n' (np.ndarray): Calculated carrier density dataset matrix.
                - 'phi' (np.ndarray, optional): Local electrostatic potential array.
                - 'title' (str, optional): Main figure title parameter text.
                - 'x_exp' (np.ndarray, optional): Experimental spatial coordinate markers.
                - 'n_exp' (np.ndarray, optional): Experimental raw carrier metrics.
                - 'exp_label' (str, optional): Custom legend identity string for raw data.

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
        title_text = kwargs.get("title")

        # Extract optional experimental configuration boundaries
        x_exp = kwargs.get("x_exp")
        n_exp = kwargs.get("n_exp")
        exp_label = kwargs.get("exp_label", "Experimental baseline")

        fig, ax1 = plt.subplots(figsize=self.fig_size)

        # Optional Title Placement
        if title_text:
            ax1.set_title(title_text, pad=10)

        # Primary axis: Carrier Concentration (Model Simulation)
        color_n = "#1f77b4"  # Scientific blue
        line1 = ax1.plot(
            x,
            n,
            color=color_n,
            linestyle="-",
            linewidth=1.5,
            label="Simulated density ($n$)",
        )
        ax1.set_xlabel("Normalised position ($x / L$)")
        ax1.set_ylabel("Normalised density ($n / N_0$)", color=color_n)
        ax1.tick_params(axis="y", labelcolor=color_n)

        lines = line1

        # Optional overlay: Experimental Scattered Data Track
        if x_exp is not None and n_exp is not None:
            # Render using marker dots to differentiate raw measurements from line simulations
            scatter_exp = ax1.scatter(
                x_exp.flatten(),
                n_exp.flatten(),
                color="#2ca02c",  # Distinct scientific green
                marker="o",
                s=20,
                edgecolors="black",
                linewidths=0.5,
                label=exp_label,
            )
            # Re-wrap references into our legend compilation list tracking
            lines = lines + [scatter_exp]

        # Secondary axis: Electrostatic Potential
        if phi is not None:
            phi = phi.flatten()
            ax2 = ax1.twinx()
            color_phi = "#d62728"  # Scientific red
            line2 = ax2.plot(
                x,
                phi,
                color=color_phi,
                linestyle="--",
                linewidth=1.5,
                label="Potential ($\phi$)",  # noqa: W605
            )
            ax2.set_ylabel("Electrostatic potential, $\phi$ (V)", color=color_phi)  # noqa: W605
            ax2.tick_params(axis="y", labelcolor=color_phi)
            ax2.grid(False)
            lines = lines + line2

        # Compile and place a unified legend box
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        plt.savefig(save_path, dpi=300)
        plt.close(fig)
