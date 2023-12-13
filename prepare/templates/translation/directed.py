from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Translate from {source_language} to {target_language}: {text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.translation.directed.simple",
        ]
    ),
    "templates.translation.directed.all",
    overwrite=True,
)
