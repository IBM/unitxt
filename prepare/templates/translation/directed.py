from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Translate from {source_language} to {target_language}: {text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.simple",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Translate the provided text from {source_language} to {target_language}, ensuring precision and "
        "maintaining formal language standards: {text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.formal",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Hey, could you help me translate this cool text from {source_language} to {target_language}?\n"
        "{text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.casual",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Time for a translation adventure! Take this text from {source_language} to {target_language} "
        "and add a dash of playfulness. Let's make it sparkle!\n Text: {text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.playful",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Execute a precise translation of the text from {source_language} to {target_language}. Please "
        "ensure accuracy and clarity, adhering to grammatical conventions and idiomatic expressions in "
        "the target language.\n Text: {text}",
        output_format="{translation}",
    ),
    "templates.translation.directed.instructional",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.translation.directed.simple",
            "templates.translation.directed.formal",
            "templates.translation.directed.casual",
            "templates.translation.directed.playful",
            "templates.translation.directed.instructional",
        ]
    ),
    "templates.translation.directed.all",
    overwrite=True,
)
