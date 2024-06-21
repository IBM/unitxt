from unitxt.catalog import add_to_catalog
from unitxt.templates import (
    InputOutputTemplate,
    TemplatesList,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Summarize the following {document_type}: {document}.",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.full",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Summarize the following {document_type}.",
        input_format="{document_type}:\n{document}\nSummary:\n",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.instruct_full",
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
        instruction="Summarize the following text into one sentence.",
        input_format="Text:\n{document}\nSummary:\n",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.instruct_one_sentence",
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
        instruction="The following {document_type} is to be summarized into one sentence.",
        input_format="{document_type}:\n{document}\nSummary:\n",
        output_format="{summary}",
        postprocessors=[
            "processors.take_first_non_empty_line",
        ],
    ),
    "templates.summarization.abstractive.instruct_passive",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Write a succinct summary of the following {document_type}: {document}.",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.write_succinct",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Write a succinct summary of the following {document_type}.",
        input_format="{document_type}:\n{document}\nSummary:\n",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.instruct_write_succinct",
    overwrite=True,
)


add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n Text: {document}",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.formal",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Produce a succinct summary for the following text, extracting the fundamental concepts and "
        "crucial information.\n{document}",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.formal_without_label",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Sum up the text with a quick overview, pulling out the main ideas and important details.\n"
        "Text: {document}",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.casual",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Craft a brief summary for the supplied text, distilling the essential concepts and vital "
        "information.\nText: {document}",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.professional",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="Guide the creation of a concise summary for the provided text, carefully "
        "extracting the central ideas and imperative information.\nText: {document}",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.instructive",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Summarize the following {document_type}.",
        input_format="{document_type}:\n{document}.",
        target_prefix="Summary:\n",
        output_format="{summary}",
        title_fields=["document_type"],
    ),
    "templates.summarization.abstractive.title",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="TL;DR:",
        input_format="{document}\nSummary:",
        output_format="{summary}",
    ),
    "templates.summarization.abstractive.instruct_tldr",
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
            "templates.summarization.abstractive.full",
            "templates.summarization.abstractive.one_sentence",
            "templates.summarization.abstractive.passive",
            "templates.summarization.abstractive.write_succinct",
            "templates.summarization.abstractive.title",
            "templates.summarization.abstractive.instruct_full",
            "templates.summarization.abstractive.instruct_one_sentence",
            "templates.summarization.abstractive.instruct_passive",
            "templates.summarization.abstractive.instruct_write_succinct",
            "templates.summarization.abstractive.instruct_tldr",
        ]
    ),
    "templates.summarization.abstractive.all",
    overwrite=True,
)
