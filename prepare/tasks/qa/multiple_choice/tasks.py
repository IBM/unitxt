from typing import List, Union

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.types import Audio, Dialog, Image, Table, Text

add_to_catalog(
    Task(
        input_fields={
            "context": Union[Text, Image, Audio, Table, Dialog],
            "context_type": str,
            "question": str,
            "choices": List[str],
        },
        reference_fields={"answer": Union[int, str], "choices": List[str]},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context",
    overwrite=True,
)


add_to_catalog(
    Task(
        input_fields={"topic": str, "question": str, "choices": List[str]},
        reference_fields={"answer": Union[int, str], "choices": List[str]},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_topic",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={"question": str, "choices": List[str]},
        reference_fields={"answer": Union[int, str], "choices": List[str]},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.open",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={
            "topic": str,
            "context": Union[Text, Image, Audio, Table, Dialog],
            "context_type": str,
            "question": str,
            "choices": List[str],
        },
        reference_fields={"answer": Union[int, str], "choices": List[str]},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    "tasks.qa.multiple_choice.with_context.with_topic",
    overwrite=True,
)
