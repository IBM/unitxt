from typing import List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"context": str, "context_type": str, "question": str},
        outputs={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.squad"],
    ),
    "tasks.qa.with_context.extractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs={"context": str, "context_type": str, "question": str},
        outputs={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    "tasks.qa.with_context.abstractive",
    overwrite=True,
)

add_to_catalog(
    Task(
        inputs={"question": str},
        outputs={"answers": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
    ),
    "tasks.qa.open",
    overwrite=True,
)
