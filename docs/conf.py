# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys
from dataclasses import Field as _Field

from unitxt.artifact import Artifact
from unitxt.dataclass import Dataclass, Field, fields, get_field_default
from unitxt.settings_utils import get_constants

constants = get_constants()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import create_catalog_docs

create_catalog_docs()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unitxt"
copyright = "2023, IBM Research"
author = "IBM Research"
release = constants.version
html_short_title = "Unitxt"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxext.opengraph",
    "sphinx.ext.linkcode",
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
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_show_sphinx = False
html_favicon = "./static/favicon.ico"
html_title = "Unitxt"

ogp_image = (
    "https://raw.githubusercontent.com/IBM/unitxt/main/docs/static/opg_image.png"
)

autodoc_default_flags = [
    "members",
    "private-members",
    "special-members",
    #'undoc-members',
    "show-inheritance",
]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    try:
        # Import the module
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        # Get the source file and line number
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    if not fn or not lineno:
        return None

    # Convert file path to relative path
    fn = os.path.relpath(fn, start=os.path.dirname(__file__))
    # Construct the GitHub URL
    return f"https://github.com/IBM/unitxt/blob/main/src/{fn}#L{lineno}"


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


def process_init_signature(app, what, name, obj, options, signature, return_annotation):
    # Check if the object is a class inheriting from Artifact
    if what == "class" and issubclass(obj, Dataclass):
        # Check if the current signature is for the __init__ method

        params = []
        for field in fields(obj):
            if not field.name.startswith("__"):
                new_type = (
                    str(field.type)
                    .replace("typing.", "")
                    .replace("<class '", "")
                    .replace("'>", "")
                )
                param = f"{field.name}: {new_type}"
                default = get_field_default(field)
                if "MISSING_TYPE" not in repr(default):
                    if isinstance(default, Artifact):
                        default = default.__id__
                    param += f" = {default!r}"
                else:
                    param += " = __required__"
                params.append(param)
        new_signature = f"({', '.join(params)})"
        return new_signature, return_annotation
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", process_init_signature)
    app.connect("autodoc-skip-member", autodoc_skip_member)
