from typing import List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"context": str, "context_type": str, "question": str},
        reference_fields={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.squad"],
    ),
    "tasks.qa.with_context.extractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={"context": str, "context_type": str, "question": str},
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
