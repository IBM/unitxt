from unitxt.catalog import add_to_catalog
from unitxt.templates import (
    InputOutputTemplate,
    MultiReferenceTemplate,
    TemplatesList,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Question: {question}",
        references_field="answers",
        target_prefix="Answer: ",
    ),
    "templates.qa.open.simple",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question.",
        input_format="Question: {question}",
        target_prefix="Answer: ",
        references_field="answers",
    ),
    "templates.qa.open.simple2",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question.",
        input_format="Question:\n{question}",
        target_prefix="Answer:\n",
        references_field="answers",
    ),
    "templates.qa.open.title",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question.",
        input_format="Question:\n{question}",
        target_prefix="Answer:\n",
        references_field="answers",
    ),
    "templates.qa.open",
    overwrite=True,
)

# empty qa template
add_to_catalog(
    InputOutputTemplate(
        input_format="{question}",
        output_format="{answers}",
    ),
    "templates.qa.open.empty",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.open",
            "templates.qa.open.simple",
            "templates.qa.open.simple2",
            "templates.qa.open.title",
            "templates.qa.open.empty",
        ]
    ),
    "templates.qa.open.all",
    overwrite=True,
)
