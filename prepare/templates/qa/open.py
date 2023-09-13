from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(input_format="Question: {question}", output_format="{answer}"),
    "templates.qa.open.simple",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(input_format="answer the question: {question}", output_format="{answer}"),
    "templates.qa.open.simple2",
    overwrite=True,
)
