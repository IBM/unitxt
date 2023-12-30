from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Classify the {type_of_class} of following {text_type} to one of these options: {classes}. Text: {text}",
        output_format="{label}",

        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.default",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="{text}",
        output_format="{label}",
    ),
    "templates.classification.multi_class.empty",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.classification.multi_class.default",
            "templates.classification.multi_class.empty",
        ]
    ),
    "templates.classification.multi_class.all",
    overwrite=True,
)
