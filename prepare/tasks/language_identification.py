from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"text": "str"},
        reference_fields={"label": "str"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.language_identification",
    overwrite=True,
)
