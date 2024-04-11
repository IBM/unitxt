from typing import Tuple

from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["question", "prediction"],
        outputs=["output"],
        metrics=["metrics.spearman"],
    ),
    "tasks.llm_as_judge.model_response_assessment",
    overwrite=True,
)
