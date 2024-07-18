from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "context": "str",
            "context_type": "str",
            "question": "str",
            "choices": "List[str]",
        },
        reference_fields={"answer": "Union[int,str]", "choices": "List[str]"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context",
    overwrite=True,
)


add_to_catalog(
    Task(
        input_fields={"topic": "str", "question": "str", "choices": "List[str]"},
        reference_fields={"answer": "Union[int,str]", "choices": "List[str]"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_topic",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={"question": "str", "choices": "List[str]"},
        reference_fields={"answer": "Union[int,str]", "choices": "List[str]"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.open",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={
            "topic": "str",
            "context": "str",
            "context_type": "str",
            "question": "str",
            "choices": "List[str]",
        },
        reference_fields={"answer": "Union[int,str]", "choices": "List[str]"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context.with_topic",
    overwrite=True,
)
