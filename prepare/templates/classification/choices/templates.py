from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(input_format="Text: {text}, Choices: {choices}.", output_format="{label}"),
    "templates.classification.choices.simple",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Given this sentence: {text}, classify if it is {choices}.",
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

add_to_catalog(
    TemplatesList(
        [
            "templates.classification.choices.simple",
            "templates.classification.choices.simple2",
            "templates.classification.choices.informed",
        ]
    ),
    "templates.classification.choices.all",
    overwrite=True,
)
