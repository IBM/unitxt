from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.types import Audio

add_to_catalog(
    Task(
        input_fields={
            "audio": Audio,
            "target_language": str,
        },
        reference_fields={"translation": str},
        prediction_type=str,
        metrics=["metrics.normalized_sacrebleu"],
    ),
    "tasks.translation.speech",
    overwrite=True,
)
