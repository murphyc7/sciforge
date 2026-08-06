"""Abstract base interfaces for extensible scientific visualisation plotters.

Enforces unified execution lifecycles for setting up canvases, drawing data curves,
and exporting vector graphics.
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from sciforge.visualisation.core_theme import get_academic_theme


class BasePlotter(ABC):
    """Abstract infrastructure layer for generating publication-quality figures."""

    def __init__(self, fig_size: tuple[float, float] = (4.5, 3.5)) -> None:
        """Initialises the plotting canvas with precise dimension scales.

        Args:
            fig_size (tuple[float, float], optional): Physical dimensions in inches.
                Defaults to (4.5, 3.5), matching standard single-column journal width.
        """
        self.fig_size = fig_size
        # Apply the global structural styling rules
        plt.rcParams.update(get_academic_theme())

    @abstractmethod
    def render(self, save_path: str, **kwargs: any) -> None:
        """Executes data plotting operations and saves the output to a file.

        Args:
            save_path (str): File destination path (supports .pdf, .svg, .png).
            **kwargs: Flexible keyword arguments representing dataset targets.
        """
        pass
