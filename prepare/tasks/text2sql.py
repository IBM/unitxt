from typing import Optional

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.types import SQLDatabase

add_to_catalog(
    Task(
        input_fields={
            "id": str,
            "utterance": str,
            "hint": Optional[str],
            "db": SQLDatabase,
        },
        reference_fields={"query": str},
        prediction_type=str,
        metrics=["metrics.text2sql.execution_accuracy", "metrics.anls"],
    ),
    "tasks.text2sql",
    overwrite=True,
)
