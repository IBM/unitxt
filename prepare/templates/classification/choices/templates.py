from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(input_format="Text: {text}, Choices: {choices}.", output_format="{label}"),
    "templates.classification.choices.simple",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Given this sentence: {sentence}, classify if it is {choices}.",
        output_format="{label}",
    ),
    "templates.classification.choices.simple2",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Classify the follwoing text to one of the options: {choices}, Text: {text}",
        output_format="{label}",
    ),
    "templates.classification.choices.informed",
    overwrite=True,
)
