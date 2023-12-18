from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    "tasks.qa.contextual",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    "tasks.qa.open",
    overwrite=True,
)
