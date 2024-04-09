from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "source_language": "str", "target_language": "str"},
        outputs={"translation": "str"},
        prediction_type="str",
        metrics=["metrics.normalized_sacrebleu"],
    ),
    "tasks.translation.directed",
    overwrite=True,
)
