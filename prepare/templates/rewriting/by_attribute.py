from src.unitxt import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        instruction="Rewrite the given {input_text_type} according to the required attribute.",
        input_format="Rewrite this {input_text_type} into more {required_attribute} {output_text_type}.\nThe {input_text_type}: {input_text}",
        target_prefix="More {required_attribute} {output_text_type}: ",
        output_format="{output_text}",
    ),
    "templates.rewriting.by_attribute.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.rewriting.by_attribute.default",
        ]
    ),
    "templates.rewriting.by_attribute.all",
    overwrite=True,
)
