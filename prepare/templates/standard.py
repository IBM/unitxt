from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import KeyValTemplate

add_to_catalog(
    KeyValTemplate(),
    "templates.key_val",
    overwrite=True,
)

add_to_catalog(
    KeyValTemplate(pairs_seperator="\n", use_keys_for_outputs=True),
    "templates.key_val_with_new_lines",
    overwrite=True,
)

add_to_catalog(
    KeyValTemplate(use_keys_for_inputs=False),
    "templates.empty",
    overwrite=True,
)
