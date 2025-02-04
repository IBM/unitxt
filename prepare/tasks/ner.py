from typing import List, Tuple

from unitxt.blocks import Task
from unitxt.catalog import add_link_to_catalog, add_to_catalog

add_to_catalog(
    Task(
        input_fields={"text": str, "entity_type": str},
        reference_fields={
            "spans_starts": List[int],
            "spans_ends": List[int],
            "text": str,
            "labels": List[str],
        },
        prediction_type=List[Tuple[str, str]],
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.ner.single_entity_type",
    overwrite=True,
)

add_link_to_catalog(
    artifact_linked_to="tasks.span_labeling.extraction",
    name="tasks.ner.all_entity_types",
    deprecate=False,
    overwrite=True,
)
