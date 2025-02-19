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
        instruction="Given the following {type_of_input_a} and {type_of_input_b}, generate the corresponding {type_of_output}."
        + "\nHere are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.",
        input_format="{type_of_input_a}: \n{input_a} \n{type_of_input_b}: \n{input_b} \n{type_of_output}:",
        output_format="{output}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.generation.from_pair.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(["templates.generation.from_pair.default"]),
    "templates.generation.from_pair.all",
    overwrite=True,
)
