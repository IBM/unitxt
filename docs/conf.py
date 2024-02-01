# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from dataclasses import Field as _Field

from unitxt.artifact import Artifact
from unitxt.dataclass import Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import CatalogDocsBuilder

CatalogDocsBuilder().run()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unitxt"
copyright = "2023, IBM Research"
author = "IBM Research"
release = "1.0.0"
html_short_title = "Unitxt"

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

html_theme = "piccolo_theme"
html_logo = "./static/logo.png"
html_theme_options = {
    "show_theme_credit": False,
    "source_url": "https://github.com/IBM/unitxt/",
    "theme_color": "pink",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_show_sphinx = False
html_favicon = "./static/favicon.ico"

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
