# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the src directory to the path so Sphinx can find nvbenjo
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Nvbenjo"
copyright = "2026, lukas-jkl"
author = "lukas-jkl"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy/Google style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Napoleon settings for better NumPy-style formatting
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "TensorLike": "nvbenjo.utils.TensorLike",
    "array_like": ":term:`array_like`",
}
napoleon_attr_annotations = True

# Add NumPy intersphinx for better links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/lukas-jkl/nvbenjo",
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
}
# Remove left sidebar (don't need it for now)
html_sidebars = {"**": []}
html_show_sourcelink = False
html_copy_source = False

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "member-order": "bysource",
    "members": False,
    "undoc-members": False,
    "exclude-members": "__weakref__, __init__",
    "inherited-members": False,
}

# Show type hints in signature only
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "mixed"
