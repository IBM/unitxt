from unitxt import add_to_catalog
from unitxt.templates import (
    SpanLabelingTemplate,
    TemplatesList,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, extract the objects for which the entity type expressed is one of {entity_types}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.extract",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, extract spans having a entity type: {entity_types}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.having",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}\nFrom this {text_type}, extract entities that carry one of the following types: {entity_types}.",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.carry",
    overwrite=True,
)

add_to_catalog(
    SpanLabelingTemplate(
        input_format="{text_type}: {text}",
        instruction="From the following {text_type}, identify spans with entity type:{entity_types}.",
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
        instruction="From the following {text_type}, extract the objects for which the entity type expressed is one of {entity_types}.",
        target_prefix="entity type:\n",
        postprocessors=["processors.to_span_label_pairs"],
        title_fields=["text_type"],
    ),
    "templates.span_labeling.extraction.title",
    overwrite=True,
)


add_to_catalog(
    SpanLabelingTemplate(
        instruction="""From the given {text_type}, extract all the entities of the following entity types: {entity_types}.
Return the output in this exact format:
The output should be a comma separated list of pairs of entity and corresponding entity_type.
Use a colon to separate between the entity and entity_type. """,
        input_format="{text_type}:\n{text}",
        postprocessors=["processors.to_span_label_pairs"],
    ),
    "templates.span_labeling.extraction.detailed",
    overwrite=True,
)


add_to_catalog(
    TemplatesList(
        items=[
            "templates.span_labeling.extraction.detailed",
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
