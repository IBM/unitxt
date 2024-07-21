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
    "templates.rag.response_generation.please_respond",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question, basing your answer on the context",
        input_format="Context: {contexts}\nQuestion: {question}.\n",
        target_prefix="Answer:",
        references_field="reference_answers",
    ),
    "templates.rag.response_generation.answer_based_on_context",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question, basing your answer on the context",
        input_format="Question: {question}.\nContext: {contexts}\n",
        target_prefix="Answer:",
        references_field="reference_answers",
    ),
    "templates.rag.response_generation.answer_based_on_context_inverted",
    overwrite=True,
)
