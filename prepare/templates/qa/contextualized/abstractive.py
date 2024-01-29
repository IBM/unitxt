from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Context: {context}\nQuestion: {question}",
        references_field="answer",
    ),
    "templates.qa.contextualized.abstractive.simple",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Question: {question}\nContext: {context}",
        references_field="answer",
    ),
    "templates.qa.contextualized.abstractive.simple_question_first",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="based on this text: {context}\n answer the question: {question}",
        references_field="answer",
    ),
    "templates.qa.contextualized.abstractive.simple2",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Given the context provided in the passage, answer the following question: {question} \nContext: "
        "{context}",
        references_field="answer",
    ),
    "templates.qa.contextualized.abstractive.simple3",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Provide a concise and accurate response to the question: {question}, based on the information "
        "available in the passage: {context}",
        references_field="answer",
    ),
    "templates.qa.contextualized.abstractive.simple4",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.contextualized.abstractive.simple",
            "templates.qa.contextualized.abstractive.simple_question_first",
            "templates.qa.contextualized.abstractive.simple2",
            "templates.qa.contextualized.abstractive.simple3",
            "templates.qa.contextualized.abstractive.simple4",
        ]
    ),
    "templates.qa.contextualized.abstractive.all",
    overwrite=True,
)
