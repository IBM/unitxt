from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "text": "str",
            "text_type": "str",
            "class_type": "str",
            "classes": "List[str]",
        },
        outputs={
            "text": "List[str]",
            "spans_starts": "List[str]",
            "spans_ends": "List[str]",
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
