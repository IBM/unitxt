from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        instruction="""You are given a text. In what language is this text written?""",
        input_format="Text: {text}",
        output_format="{label}",
        target_prefix="The text is in ",
        postprocessors=[
            "processors.take_first_word",
        ],
    ),
    "templates.language_identification.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.language_identification.simple",
        ]
    ),
    "templates.language_identification.all",
    overwrite=True,
)
