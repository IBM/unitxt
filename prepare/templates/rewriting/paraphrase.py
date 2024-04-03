from src.unitxt import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        instruction="Rewrite the given {text_type} differently.",
        input_format="{text_type}: {input_text}",
        target_prefix="Paraphrase: ",
        output_format="{output_text}",
        title_fields=["text_type"],
    ),
    "templates.rewriting.paraphrase.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.rewriting.paraphrase.default",
        ]
    ),
    "templates.rewriting.paraphrase.all",
    overwrite=True,
)
