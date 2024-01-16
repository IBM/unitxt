from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    "tasks.qa.contextual.extractive",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["context", "question"],
        outputs=["answer"],
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    "tasks.qa.contextual.abstractive",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["question"],
        outputs=["answer"],
        metrics=["metrics.rouge"],
    ),
    "tasks.qa.open",
    overwrite=True,
)
