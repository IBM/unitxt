from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.completion.multiple_choice",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "completion_type"],
        outputs=["completion"],
        metrics=["metrics.rouge"],
    ),
    "tasks.completion.abstractive",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "completion_type"],
        outputs=["completion"],
        metrics=["metrics.squad"],
    ),
    "tasks.completion.extractive",
    overwrite=True,
)
