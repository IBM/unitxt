from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["context", "context_type", "question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context",
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
        inputs=["topic", "context", "context_type", "question", "choices"],
        outputs=["answer", "choices"],
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context.with_topic",
    overwrite=True,
)
