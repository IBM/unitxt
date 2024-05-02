from unitxt.catalog import add_to_catalog
from unitxt.templates import OutputQuantizingTemplate, TemplatesList

add_to_catalog(
    OutputQuantizingTemplate(
        input_format="""
                   Given this text: '{text}', on a scale of {min_value} to {max_value}, what is the {attribute_name} of this text?
                """.strip(),
        output_format="{attribute_value}",
        quantum=0.2,
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.cast_to_float_return_zero_if_failed",
        ],
    ),
    "templates.regression.single_text.simple",
    overwrite=True,
)

add_to_catalog(
    OutputQuantizingTemplate(
        instruction="Given a text, on a scale of {min_value} to {max_value}, what is the {attribute_name} of this text?",
        input_format="Text:\n{text}",
        output_format="{attribute_value}",
        target_prefix="{attribute_name}:\n",
        quantum=0.2,
        title_fields=["attribute_name"],
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.cast_to_float_return_zero_if_failed",
        ],
    ),
    "templates.regression.single_text.title",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.regression.single_text.simple",
            "templates.regression.single_text.title",
        ]
    ),
    "templates.regression.single_text.all",
    overwrite=True,
)
