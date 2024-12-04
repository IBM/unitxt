from typing import List, Union

from unitxt.blocks import Task
from unitxt.catalog import add_link_to_catalog, add_to_catalog
from unitxt.types import Audio, Dialog, Image, Table, Text

add_to_catalog(
    Task(
        __deprecated_msg__="This task is no longer used and was merged with 'tasks.qa.with_context'. It should be replaced with 'tasks.qa.with_context[metrics=[metrics.squad]]' to get the same behavior.",
        input_fields={
            "context": Union[Text, Table, Dialog],
            "context_type": str,
            "question": str,
        },
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.squad"],
    ),
    "tasks.qa.with_context.extractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is the Question Answering Task with provided context (which is a either text, image, audio, table , or dialog).
The 'tasks.qa.open' should be used if there is no context.  One or more ground truth answers can be provided in the 'answers' field.
By default, classifical Rouge metric is used , but list of additional applicable metrics can be found under 'metrics.qa' in the Unitxt catalog.
        """,
        input_fields={
            "context": Union[Text, Image, Audio, Table, Dialog],
            "context_type": str,
            "question": str,
        },
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
        defaults={"answers": []},
        default_template=["templates.qa.with_context.title"],
    ),
    "tasks.qa.with_context",
    overwrite=True,
)

add_link_to_catalog(
    artifact_linked_to="tasks.qa.with_context",
    name="tasks.qa.with_context.abstractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is the Question Answering Task composed of question answer pair , without provided context.
            The 'tasks.qa.with_context' should be used if there is no context.
            By default, classifical Rouge metric is used , but list of additional applicable metrics can be found under 'metrics.qa' in the Unitxt catalog.
            """,
        input_fields={"question": str},
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
        defaults={"answers": []},
        default_template=["templates.qa.open.title"],
    ),
    "tasks.qa.open",
    overwrite=True,
)
