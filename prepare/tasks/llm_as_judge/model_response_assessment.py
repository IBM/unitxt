from typing import Tuple

from src.unitxt.blocks import FormTask
from src.unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["model_input", "model_output", "model_input_type", "model_output_type"],
        outputs=["score", "reason"],
        prediction_type=Tuple[int, str],
        metrics=["metrics.spearman"],
        augmentable_inputs=["input", "output"],
    ),
    "tasks.llm_as_judge.model_response_assessment",
    overwrite=True,
)
