from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "text": "str",
            "text_type": "str",
            "class_type": "str",
            "classes": "List[str]",
        },
        reference_fields={
            "text": "str",
            "spans_starts": "List[int]",
            "spans_ends": "List[int]",
            "labels": "List[str]",
        },
        prediction_type="List[Tuple[str,str]]",
        metrics=[
            "metrics.ner",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "class_type": "entity type"},
    ),
    "tasks.span_labeling.extraction",
    overwrite=True,
)
