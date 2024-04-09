from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "class_type", "classes"],
        outputs=["text", "spans_starts", "spans_ends", "labels"],
        metrics=[
            "metrics.ner",
        ],
        augmentable_inputs=["text"],
    ),
    "tasks.span_labeling.extraction",
    overwrite=True,
)
