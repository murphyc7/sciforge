"""Global visualization theme and style overrides for scientific publishing.

Configures matplotlib typography, line densities, and layout bounds to comply
with standard peer-reviewed journal constraints.
"""

from typing import Any


def get_academic_theme() -> dict[str, Any]:
    """Returns a dictionary of strict stylesheet parameters for scientific plotting.

    Employs readable sans-serif fallback hierarchies, optimised line weights,
    and crisp vector boundaries suitable for high-resolution printing formats.
    """
    return {
        "text.usetex": False,  # True if your system has a live local LaTeX installation
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "none",
        "figure.dpi": 300,  # Publication quality resolution baseline
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
