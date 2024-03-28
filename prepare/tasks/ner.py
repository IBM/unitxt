from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "entity_type": "str"},
        outputs={
            "spans_starts": "List[int]",
            "spans_ends": "List[int]",
            "text": "List[str]",
            "labels": "List[str]",
        },
        prediction_type="List[Tuple[str,str]]",
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.ner.single_entity_type",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"text": "str", "entity_types": "List[str]"},
        outputs={
            "spans_starts": "List[int]",
            "spans_ends": "List[int]",
            "text": "List[str]",
            "labels": "List[str]",
        },
        prediction_type="List[Tuple[str,str]]",
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.ner.all_entity_types",
    overwrite=True,
)
