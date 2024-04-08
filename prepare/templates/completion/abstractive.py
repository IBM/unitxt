from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="{context}",
        output_format="{completion}",
    ),
    "templates.completion.abstractive.empty",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Write the best {completion_type} to the {context_type}.",
        input_format="{context}",
        output_format="{completion}",
    ),
    "templates.completion.abstractive.standard",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.completion.abstractive.empty",
            "templates.completion.abstractive.standard",
        ]
    ),
    "templates.completion.abstractive.all",
    overwrite=True,
)
