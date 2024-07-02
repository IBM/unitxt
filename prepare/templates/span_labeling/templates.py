from unitxt import add_to_catalog
from unitxt.templates import (
    SpanLabelingTemplate,
    TemplatesList,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, extract the objects for which the {class_type} expressed is one of {classes}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.extract",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, extract spans having a {class_type}: {classes}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.having",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}\nFrom this {text_type}, extract entities that carry one of the following types: {classes}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.carry",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, identify spans with {class_type}:{classes}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.identify",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text}",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.empty",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}:\n{text}",
        instruction="From the following {text_type}, extract the objects for which the {class_type} expressed is one of {classes}.\n\n",
        target_prefix="{class_type}:\n",
        postprocessors=["processors.to_span_label_pairs"],
        title_fields=["text_type", "class_type"],
    ),
    "templates.span_labeling.extraction.title",
    overwrite=True,
)


add_to_catalog(
    TemplatesList(
        items=[
            "templates.span_labeling.extraction.extract",
            "templates.span_labeling.extraction.having",
            "templates.span_labeling.extraction.carry",
            "templates.span_labeling.extraction.identify",
            "templates.span_labeling.extraction.title",
            "templates.span_labeling.extraction.empty",
        ]
    ),
    "templates.span_labeling.extraction.all",
    overwrite=True,
)
