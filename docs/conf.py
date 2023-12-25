# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from dataclasses import Field as _Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import build_catalog_rst

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

autodoc_default_flags = [
    "members",
    "private-members",
    "special-members",
    #'undoc-members',
    "show-inheritance",
]


def autodoc_skip_member(app, what, name, obj, would_skip, options):
    if would_skip:
        return True

    if isinstance(obj, (Field, _Field, bool, int, str, float)):
        return True

    if obj is None or type(obj) is object:
        return True

    if hasattr(obj, "__qualname__"):
        class_name = obj.__qualname__.split(".")[0]
        if (
            class_name
            and Artifact.is_registered_class_name(class_name)
            and class_name != name
        ):
            return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
