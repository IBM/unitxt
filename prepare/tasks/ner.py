from unitxt import add_to_catalog
from unitxt.blocks import FormTask

add_to_catalog(
    FormTask(
        inputs=["text", "entity_type"],
        outputs=["spans_starts", "spans_ends", "text", "labels"],
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.ner.single_entity_type",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text", "entity_types"],
        outputs=["spans_starts", "spans_ends", "text", "labels"],
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.ner.all_entity_types",
    overwrite=True,
)
