from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["input", "type_of_input", "type_of_output"],
        outputs=["output"],
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
    ),
    "tasks.generation",
    overwrite=True,
)
