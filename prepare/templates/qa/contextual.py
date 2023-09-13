from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(input_format="Context: {context}\nQuestion: {question}", output_format="{answer}"),
    "templates.qa.contextual.simple",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="based on this text: {context}\n answer the question: {question}", output_format="{answer}"
    ),
    "templates.qa.contextual.simple2",
    overwrite=True,
)
