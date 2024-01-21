from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import OutputQuantizingTemplate, TemplatesList

add_to_catalog(
    OutputQuantizingTemplate(
        input_format="""
                   Given this text: '{text1}' and this text: '{text2}', on a scale of {min_value} to {max_value}, what is the {type_of_value} of this texts?
                """.strip(),
        output_format="{value}",
        quantum=0.2,
    ),
    "templates.regression.bounded.pair.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.regression.bounded.pair.simple",
        ]
    ),
    "templates.regression.bounded.pair.all",
    overwrite=True,
)
