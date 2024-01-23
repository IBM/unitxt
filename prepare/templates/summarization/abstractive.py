from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format="Summarize the following {document_type}: {document}.",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.full",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Summarize the following text into one sentence: {document}.",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.one_sentence",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="The following {document_type} is to be summarized into one sentence: {document}.",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.passive",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Write a succinct summary of the following {document_type}: {document}.",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.write_succinct",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.summarization.abstractive.full",
            "templates.summarization.abstractive.one_sentence",
            "templates.summarization.abstractive.passive",
            "templates.summarization.abstractive.write_succinct",
        ]
    ),
    "templates.summarization.abstractive.all",
    overwrite=True,
)
