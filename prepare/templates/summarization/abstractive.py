from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n Text: {text}",
        output_format="{target}",
    ),
    "templates.summarization.abstractive.formal",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n{text}",
        output_format="{target}",
    ),
    "templates.summarization.abstractive.formal_without_label",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Sum up the text with a quick overview, pulling out the main ideas and important details.\n"
        "Text: {text}",
        output_format="{target}",
    ),
    "templates.summarization.abstractive.casual",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Craft a brief summary for the supplied text, distilling the essential concepts and vital "
        "information.\nText: {text}",
        output_format="{target}",
    ),
    "templates.summarization.abstractive.professional",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Guide the creation of a concise summary for the provided text, carefully "
        "extracting the central ideas and imperative information.\nText: {text}",
        output_format="{target}",
    ),
    "templates.summarization.abstractive.instructive",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.summarization.abstractive.formal",
            "templates.summarization.abstractive.formal_without_label",
            "templates.summarization.abstractive.casual",
            "templates.summarization.abstractive.professional",
            "templates.summarization.abstractive.instructive",
        ]
    ),
    "templates.summarization.abstractive.all",
    overwrite=True,
)
