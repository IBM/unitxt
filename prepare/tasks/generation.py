from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["input", "type_of_input"],
        outputs=["output"],
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
    ),
    "tasks.generation",
    overwrite=True,
)
