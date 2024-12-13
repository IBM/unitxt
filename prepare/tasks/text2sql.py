from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.text2sql.metrics import ExecutionAccuracy  # noqa

add_to_catalog(
    Task(
        input_fields={
            "id": int,
            "utterance": str,
            "db_id": str,
            "dbms": str,
            "evidence": str,
        },
        reference_fields={"query": str},
        prediction_type=str,
        metrics=["metrics.text2sql.execution_accuracy"],
    ),
    "tasks.text2sql",
    overwrite=True,
)
