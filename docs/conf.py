# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from catalog import build_catalog_rst

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

build_catalog_rst()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unitxt"
copyright = "2023, IBM Research"
author = "IBM Research"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "logo_only": True,
    "display_version": False,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#ff66ff",
}
suppress_warnings = [
    "app.add_node",
    "app.add_directive",
    "app.add_role",
    "app.add_generic_role",
    "app.add_source_parser",
    "autosectionlabel.*",
    "download.not_readable",
    "epub.unknown_project_files",
    "epub.duplicated_toc_entry",
    "i18n.inconsistent_references",
    "index",
    "image.not_readable",
    "ref.term",
    "ref.ref",
    "ref.numref",
    "ref.keyword",
    "ref.option",
    "ref.citation",
    "ref.footnote",
    "ref.doc",
    "ref.python",
    "misc.highlighting_failure",
    "toc.circular",
    "toc.excluded",
    "toc.not_readable",
    "toc.secnum",
]
