from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.text2sql.metrics import ExecutionAccuracy  # noqa
from unitxt.text2sql.types import SQLSchema

add_to_catalog(
    Task(
        input_fields={
            "id": int,
            "utterance": str,
            "db_id": str,
            "dbms": str,
            "evidence": str,
            "schema": SQLSchema,
        },
        reference_fields={"query": str},
        prediction_type=str,
        metrics=["metrics.text2sql.execution_accuracy", "metrics.anls"],
    ),
    "tasks.text2sql",
    overwrite=True,
)
