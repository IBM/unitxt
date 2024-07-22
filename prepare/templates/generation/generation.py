from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

### Generation

add_to_catalog(
    InputOutputTemplate(
        input_format="Given the following {type_of_input}, generate the corresponding {type_of_output}. {type_of_input}: {input}",
        output_format="{output}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.generation.default",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="{input}",
        output_format="{output}",
    ),
    "templates.generation.empty",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(["templates.generation.default", "templates.generation.empty"]),
    "templates.generation.all",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Given the following {type_of_input} and caption, generate the corresponding {type_of_output}."
        "\n{type_of_input}: \n{input} \ncaption: \n{caption} \n{type_of_output}:",
        output_format="{output}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.generation.structured.with_caption",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(["templates.generation.structured.with_caption"]),
    "templates.generation.structured",
    overwrite=True,
)
