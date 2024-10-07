# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../..'))
from pathlib import Path
CPP_DOC_PATH = Path(__file__).absolute().parent

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'stochtree'
copyright = '2024, Drew Herren'
author = 'Drew Herren'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'breathe'
]

templates_path = ['_templates']
exclude_patterns = []

# Breathe Configuration
breathe_projects = {"StochTree": str(CPP_DOC_PATH / "doxyoutput" / "xml")}
breathe_default_project = "StochTree"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
