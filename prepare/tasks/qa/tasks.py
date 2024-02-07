from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    "tasks.qa.with_context.extractive",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "question"],
        outputs=["answer"],
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    "tasks.qa.with_context.abstractive",
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
