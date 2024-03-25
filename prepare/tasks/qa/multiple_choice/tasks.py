from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "context": "str",
            "context_type": "str",
            "question": "str",
            "choices": "List[str]",
        },
        outputs={"answer": "str", "choices": "List[str]"},
        prediction_type="Any",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context",
    overwrite=True,
)


add_to_catalog(
    FormTask(
        inputs={"topic": "str", "question": "str", "choices": "List[str]"},
        outputs={"answer": "str", "choices": "List[str]"},
        prediction_type="Any",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_topic",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"question": "str", "choices": "List[str]"},
        outputs={"answer": "str", "choices": "List[str]"},
        prediction_type="Any",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.open",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={
            "topic": "str",
            "context": "str",
            "context_type": "str",
            "question": "str",
            "choices": "List[str]",
        },
        outputs={"answer": "str", "choices": "List[str]"},
        prediction_type="Any",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context.with_topic",
    overwrite=True,
)
