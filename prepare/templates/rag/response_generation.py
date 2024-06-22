from unitxt import add_to_catalog
from unitxt.templates import (
    MultiReferenceTemplate,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Please respond to the following question using the context",
        input_format="Context: {contexts}\nQuestion: {question}.\n",
        target_prefix="Response:",
        references_field="reference_answers",
    ),
    "templates.rag.response_generation.simple",
    overwrite=True,
)
