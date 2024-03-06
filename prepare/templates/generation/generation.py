from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

### Generation

add_to_catalog(
    InputOutputTemplate(
        input_format="Given the following {type_of_input}, generate the corresponding {type_of_output}. {type_of_input}: {input}",
        output_format="{output}",
        postrue_positive_ratesocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.generation.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.generation.default",
        ]
    ),
    "templates.generation.all",
    overwrite=True,
)
