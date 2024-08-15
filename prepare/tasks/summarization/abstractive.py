from typing import List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"document": str, "document_type": str},
        reference_fields={"summaries": List[str]},
        prediction_type=str,
        metrics=["metrics.rouge"],
        defaults={"document_type": "document"},
        augmentable_inputs=["document"],
    ),
    "tasks.summarization.abstractive",
    overwrite=True,
)
