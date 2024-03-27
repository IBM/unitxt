from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"input": "str", "type_of_input": "str", "type_of_output": "str"},
        outputs={"output": "str"},
        prediction_type="str",
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
    ),
    "tasks.generation",
    overwrite=True,
)
