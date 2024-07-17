from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "text": "str",
            "source_language": "str",
            "target_language": "str",
        },
        reference_fields={"translation": "str"},
        prediction_type="str",
        metrics=["metrics.normalized_sacrebleu"],
    ),
    "tasks.translation.directed",
    overwrite=True,
)
