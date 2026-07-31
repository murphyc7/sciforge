import os
import sys

############################################################
# Configuration file for the Sphinx documentation builder. #
############################################################

# Project information
project = 'SciForge'
copyright = '2026, Conal Murphy'
author = 'Conal Murphy'
release = '0.1.0'

# Point Sphinx directly to absolute source directory layout
sys.path.insert(0, os.path.abspath("../../src"))

# Add extension modules
extensions = [
    "sphinx.ext.autodoc",  # Core extension to pull docstrings from code
    "sphinx.ext.napoleon",  # Supports clean Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Adds links to the raw source code in the docs
    "sphinx.ext.mathjax",  # Renders LaTeX physics equations in the browser
]

# Switch the visual styling to the Read the Docs theme
html_theme = "sphinx_rtd_theme"

# Templates path and exclusions stay as default
templates_path = ["_templates"]
exclude_patterns = []
