from src.unitxt.templates import KeyValTemplate

from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    KeyValTemplate(),
    "templates.key_val",
    overwrite=True,
)

add_to_catalog(
    KeyValTemplate(use_keys_for_inputs=False),
    "templates.empty",
    overwrite=True,
)