from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "text": "str",
            "text_type": "str",
            "class_type": "str",
            "classes": "List[str]",
        },
        outputs={
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
    ),
    "tasks.span_labeling.extraction",
    overwrite=True,
)
