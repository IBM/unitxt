import json

from src.unitxt import add_to_catalog
from src.unitxt.templates import (
    SpanLabelingJsonTemplate,
    SpanLabelingTemplate,
    TemplatesList,
)

"""
Templates for a targeted sentiment extraction task.
"""

task_name = "targeted_sentiment_extraction"


def add_templates():
    targeted_sentiment_templates = {
        "extract_sentiment": "From the following {text_type}, extract the objects for which the sentiment expressed is positive, and the objects for which the expressed sentiment is negative, and the objects for which the expressed sentiment is neutral.\n{text_type}: {text}\n",
        "having_sentiment": "From the following {text_type}, extract entities having a sentiment: positive, negative, neutral.\n{text_type}: {text}\n",
        "carry_sentiment": "{text_type}: {text}\nFrom this {text_type}, extract entities that carry one of the following types: positive, negative, neutral.\n",
        "entities_sentiment": "From the following {text_type}, identify entities with sentiment: positive, negative, neutral.\n{text_type}: {text}\n",
        "empty": "{text}",
    }

    template_list = []
    for template_name, template_input_format in targeted_sentiment_templates.items():
        template = SpanLabelingTemplate(
            input_format=template_input_format,
            postprocessors=["processors.to_span_label_pairs"],
        )
        full_template_name = f"templates.{task_name}.{template_name}"
        template_list.append(full_template_name)
        add_to_catalog(template, full_template_name, overwrite=True)

    sentiments = ["positive", "negative", "neutral"]

    json_templates = {
        "convert_with_explicit_keys": "Convert the following text into JSON format in a single line, with the following keys:"
        + json.dumps(sentiments)
        + ". \nText: {text}",
        "convert_with_implicit_keys": "From the following {text_type}, extract entities having a sentiment: positive, negative, neutral. Output JSON format in a single line, with the sentiment types as keys \n{text_type}: {text}",
        "empty": "{text}",
    }

    for template_name, template_input_format in json_templates.items():
        template = SpanLabelingJsonTemplate(
            input_format=template_input_format,
        )
        full_template_name = f"templates.{task_name}.as_json.{template_name}"
        template_list.append(full_template_name)
        add_to_catalog(template, full_template_name, overwrite=True)

    add_to_catalog(
        TemplatesList(template_list), f"templates.{task_name}.all", overwrite=True
    )


def add_single_sentiment_templates():
    single_sentiment_templates = {
        "sentiment_extracted": "From the following {text_type}, extract the objects for which the sentiment extracted is {sentiment_class}. If there are none, output None. \n{text_type}: {text}\n",
        "having_sentiment": "From the following {text_type}, extract entities having a sentiment: {sentiment_class}. If there  none, output None. \n{text_type}: {text}\n",
        "empty": "{text}",
    }

    for sentiment in ["positive", "negative", "neutral"]:
        template_list = []
        for (
            single_sentiment_template_name,
            single_sentiment_template_input_format,
        ) in single_sentiment_templates.items():
            template = SpanLabelingTemplate(
                input_format=single_sentiment_template_input_format,
                labels_support=[sentiment],
                span_label_format="{span}",
                postprocessors=["processors.to_span_label_pairs_surface_only"],
            )
            full_template_name = (
                f"templates.{task_name}.{sentiment}.{single_sentiment_template_name}"
            )
            template_list.append(full_template_name)
            add_to_catalog(template, full_template_name, overwrite=True)
        add_to_catalog(
            TemplatesList(template_list),
            f"templates.{task_name}.{sentiment}.all",
            overwrite=True,
        )


add_templates()
add_single_sentiment_templates()
