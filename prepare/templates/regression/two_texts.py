from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import OutputQuantizingTemplate, TemplatesList

add_to_catalog(
    OutputQuantizingTemplate(
        input_format="""
                   Given this sentence: '{text1}', on a scale of {min_value} to {max_value}, what is the {attribute_name} to this text {text2}?
                """.strip(),
        output_format="{attribute_value}",
        quantum=0.2,
    ),
    "templates.regression.two_texts.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.regression.two_texts.simple",
        ]
    ),
    "templates.regression.two_texts.all",
    overwrite=True,
)
