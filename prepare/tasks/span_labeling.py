from typing import List, Tuple

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        __description__="""This is Entity Extraction task where multiple entity types are to be extracted.
The input is the 'text' and 'entity_types' to extract (e.g. ["Organization", "Location", "Person"])

By default, classical f1 metric is used, which expects a list of <entity,entity_type> pairs.
Multiple f1 score are reported, including f1_micro and f1_macro and f1 per per entity_type.".
The template's post processors must convert the model textual predictions into the expected list format.
""",
        input_fields={
            "text": str,
            "text_type": str,
            "entity_types": List[str],
        },
        reference_fields={
            "text": str,
            "spans_starts": List[int],
            "spans_ends": List[int],
            "labels": List[str],
        },
        prediction_type=List[Tuple[str, str]],
        metrics=[
            "metrics.ner",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
        default_template="templates.span_labeling.extraction.detailed",
    ),
    "tasks.span_labeling.extraction",
    overwrite=True,
)
