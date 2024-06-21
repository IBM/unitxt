from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"input": "str", "type_of_input": "str", "type_of_output": "str"},
        outputs={"output": "str"},
        prediction_type="str",
        metrics=["metrics.normalized_sacrebleu"],
        augmentable_inputs=["input"],
        defaults={"type_of_output": "Text"},
    ),
    "tasks.generation",
    overwrite=True,
)
