from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.original",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["context", "question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.contextual",
    overwrite=True,
)


add_to_catalog(
    FormTask(
        inputs=["topic", "question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_topic",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.open",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["topic", "context", "question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.contextual_with_topic",
    overwrite=True,
)
