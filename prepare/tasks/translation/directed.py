from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["text", "source_language", "target_language"],
        outputs=["translation"],
        metrics=["metrics.normalized_sacrebleu"],
    ),
    "tasks.translation.directed",
    overwrite=True,
)
