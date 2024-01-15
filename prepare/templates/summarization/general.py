from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n Text: {text}",
        output_format="{target}",
    ),
    "templates.summarization.general.formal",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n{text}",
        output_format="{target}",
    ),
    "templates.summarization.general.formal_without_label",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Sum up the text with a quick overview, pulling out the main ideas and important details.\n"
        "Text: {text}",
        output_format="{target}",
    ),
    "templates.summarization.general.casual",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Craft a brief summary for the supplied text, distilling the essential concepts and vital "
        "information.\nText: {text}",
        output_format="{target}",
    ),
    "templates.summarization.general.professional",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Guide the creation of a concise summary for the provided text, carefully "
        "extracting the central ideas and imperative information.\nText: {text}",
        output_format="{target}",
    ),
    "templates.summarization.general.instructive",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.summarization.general.formal",
            "templates.summarization.general.formal_without_label",
            "templates.summarization.general.casual",
            "templates.summarization.general.professional",
            "templates.summarization.general.instructive",
        ]
    ),
    "templates.summarization.general.all",
    overwrite=True,
)
