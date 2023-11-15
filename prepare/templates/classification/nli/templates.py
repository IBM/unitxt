from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Given this sentence: {premise}, classify if this sentence: {hypothesis} is {choices}.",
        output_format="{label}",
        postprocessors=["processors.take_first_non_empty_line"],
    ),
    "templates.classification.nli.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.classification.nli.simple",
        ]
    ),
    "templates.classification.nli.all",
    overwrite=True,
)
