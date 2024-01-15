from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Context: {context}\nQuestion: {question}",
        references_field="answers",
    ),
    "templates.qa.contextual.simple",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Question: {question}\nContext: {context}",
        references_field="answers",
    ),
    "templates.qa.contextual.simple_question_first",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="based on this text: {context}\n answer the question: {question}",
        references_field="answers",
    ),
    "templates.qa.contextual.simple2",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Given the context provided in the passage, answer the following question: {question} \nContext: "
        "{context}",
        references_field="answers",
    ),
    "templates.qa.contextual.simple3",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Provide a concise and accurate response to the question: {question}, based on the information "
        "available in the passage: {context}",
        references_field="answers",
    ),
    "templates.qa.contextual.simple4",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.contextual.simple",
            "templates.qa.contextual.simple_question_first",
            "templates.qa.contextual.simple2",
            "templates.qa.contextual.simple3",
            "templates.qa.contextual.simple4",
        ]
    ),
    "templates.qa.contextual.all",
    overwrite=True,
)
