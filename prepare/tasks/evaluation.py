from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs=["input", "input_type", "output_type", "choices", "instruction"],
        outputs=["choices", "output_choice"],
        metrics=[
            "metrics.accuracy",
        ],
        augmentable_inputs=["input", "instruction"],
    ),
    "tasks.evaluation.preference",
    overwrite=True,
)
