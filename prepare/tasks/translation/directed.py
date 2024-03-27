from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

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
