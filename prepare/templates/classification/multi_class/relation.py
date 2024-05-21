from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="{text_a_type}: {text_a}\n{text_b_type}: {text_b}",
        output_format="{label}",
        target_prefix="The {type_of_relation} class is ",
        instruction="Given a {text_a_type} and {text_b_type} classify the {type_of_relation} of the {text_b_type} to one of {classes}.",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.default",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Given this {text_a_type}: {text_a}, classify if this {text_b_type}: {text_b} is {classes}.",
        output_format="{label}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(["templates.classification.multi_class.relation.default"]),
    "templates.classification.multi_class.relation.all",
    overwrite=True,
)
