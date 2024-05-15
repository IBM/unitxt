from unitxt import add_to_catalog
from unitxt.templates import (
    MultiReferenceTemplate,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{contexts}\n\nPlease answer the following question about this context.\n\n"
        "question: {question}.\n\nanswer:",
        references_field="reference_answers",
    ),
    "templates.rag.response_generation.simple",
    overwrite=True,
)
