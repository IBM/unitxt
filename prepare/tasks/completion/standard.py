from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.completion.multiple_choice.standard",
    overwrite=True,
)
