from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Context: {context}\nQuestion: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.simple",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="based on this text: {context}\n answer the question: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.simple2",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Based on this {context_type}:\n {context}\n answer the question: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.with_type",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{question}\nAnswer based on this {context_type}:\n {context}",
        references_field="answers",
    ),
    "templates.qa.with_context.question_first",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.with_context.simple",
            "templates.qa.with_context.simple2",
            "templates.qa.with_context.with_type",
            "templates.qa.with_context.question_first",
        ]
    ),
    "templates.qa.with_context.all",
    overwrite=True,
)
