from typing import List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "id": str,
            "utterance": str,
            "hint": str,
            "schema": List[str],
        },
        reference_fields={"linked_schema": List[str]},
        prediction_type=List[str],
        metrics=["metrics.f1_macro_multi_label"],
    ),
    "tasks.schema_linking",
    overwrite=True,
)
