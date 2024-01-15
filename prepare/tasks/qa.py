from unitxt import add_to_catalog
from unitxt.blocks import FormTask

add_to_catalog(
    FormTask(
        inputs=["context", "question"],
        outputs=["answers"],
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    "tasks.qa.contextual",
    overwrite=True,
)
