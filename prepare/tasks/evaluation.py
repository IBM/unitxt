from typing import List

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "input": str,
            "input_type": str,
            "output_type": str,
            "choices": List[str],
            "instruction": str,
        },
        outputs={
            "choices": List[str],
            "output_choice": int,
        },
        metrics=[
            "metrics.accuracy",
        ],
        augmentable_inputs=["input", "instruction"],
    ),
    "tasks.evaluation.preference",
    overwrite=True,
)
