from unitxt import add_to_catalog
from unitxt.blocks import FormTask

add_to_catalog(
    FormTask(
        inputs=["input"],
        outputs=["output"],
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
    ),
    "tasks.generation",
    overwrite=True,
)
