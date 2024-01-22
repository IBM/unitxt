from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import OutputQuantizingTemplate, TemplatesList

add_to_catalog(
    OutputQuantizingTemplate(
        input_format="""
                   Given this text: '{text}', on a scale of {min_value} to {max_value}, what is the {attribute_name} of this text?
                """.strip(),
        output_format="{attribute_value}",
        quantum=0.2,
    ),
    "templates.regression.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.regression.simple",
        ]
    ),
    "templates.regression.simple",
    overwrite=True,
)
