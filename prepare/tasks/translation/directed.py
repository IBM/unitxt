from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "source_language", "target_language"],
        outputs=["translation"],
        metrics=["metrics.normalized_sacrebleu"],
    ),
    "tasks.translation.directed",
    overwrite=True,
)
