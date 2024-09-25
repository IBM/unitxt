from typing import List, Union

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.types import Audio, Dialog, Image, Table, Text

add_to_catalog(
    Task(
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
        input_fields={
            "context": Union[Text, Image, Audio, Table, Dialog],
            "context_type": str,
            "question": str,
        },
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    "tasks.qa.with_context.abstractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={"question": str},
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
    ),
    "tasks.qa.open",
    overwrite=True,
)
