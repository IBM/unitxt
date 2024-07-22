from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"text": str},
        outputs={"label": str},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    "tasks.language_identification",
    overwrite=True,
)
