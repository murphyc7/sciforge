"""Tests for murphyc7/sciforge/src/sciforge/visualisation/"""

import os

import numpy as np
import pytest

from sciforge.visualisation.components import DualAxisTransportPlotter
from sciforge.visualisation.core_theme import get_academic_theme

#################################################################
# 1. murphyc7/sciforge/src/sciforge/visualisation/core_theme.py #
#################################################################


def test_core_theme_dictionary_contains_required_academic_keys() -> None:
    """Verifies that the visualisation theme module exposes essential publishing settings."""
    theme = get_academic_theme()
    assert isinstance(theme, dict)
    assert "figure.dpi" in theme
    assert "axes.grid" in theme
    assert theme["figure.dpi"] == 300


#################################################################
# 2. murphyc7/sciforge/src/sciforge/visualisation/components.py #
#################################################################


def test_dual_axis_plotter_saves_constant_field_file_successfully(tmp_path) -> None:
    """Verifies that the plotting engine successfully exports an image file under standard 1D inputs."""
    # Arrange: Generate a temporary file target destination using pytest's isolated directory fixture
    output_file = os.path.join(tmp_path, "test_constant.png")

    np_x = np.linspace(0.0, 1.0, 50)
    np_n = np.exp(-np_x)

    plotter = DualAxisTransportPlotter()

    # Act: Trigger the rendering pipeline context
    plotter.render(save_path=output_file, x=np_x, n=np_n, title="Test 1D Constant Run")

    # Assert: Prove that the canvas successfully wrote the graphics payload to disk
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_dual_axis_plotter_overlays_coupled_and_experimental_data(tmp_path) -> None:
    """Confirms that the plotter maps dynamic experimental scatter plots and dual axes without crashes."""
    output_file = os.path.join(tmp_path, "test_coupled.png")

    np_x = np.linspace(0.0, 1.0, 40)
    np_n = np.exp(-np_x)
    np_phi = np_x * 0.5

    np_x_exp = np.linspace(0.2, 0.8, 5)
    np_n_exp = np.exp(-np_x_exp) * 0.9

    plotter = DualAxisTransportPlotter()

    # Act: Run rendering with all optional keyword properties active simultaneously
    plotter.render(
        save_path=output_file,
        x=np_x,
        n=np_n,
        phi=np_phi,
        title="Test Multi-Variable Run",
        x_exp=np_x_exp,
        n_exp=np_n_exp,
        exp_label="Lab Measured Calibration Ref",
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_plotter_raises_key_error_when_mandatory_arrays_are_omitted(tmp_path) -> None:
    """Guarantees that omitting structural parameters triggers an explicit KeyError exception validation."""
    output_file = os.path.join(tmp_path, "test_broken.png")
    plotter = DualAxisTransportPlotter()

    # Assert: Triggering rendering without passing required inputs must throw an explicit KeyError
    with pytest.raises(KeyError, match="requires explicit 'x' and 'n'"):
        plotter.render(
            save_path=output_file, x=np.array([1, 2, 3])
        )  # Missing the 'n' density tracking target
