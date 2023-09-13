from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        input_format="Given this sentence: {premise}, classify if this sentence: {hypothesis} is {choices}.",
        output_format="{label}",
    ),
    "templates.classification.nli.simple",
    overwrite=True,
)
