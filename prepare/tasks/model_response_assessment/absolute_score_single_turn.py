from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"question": "str", "model_output": "str"},
        outputs={"rating_label": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.model_response_assessment.absolute_score_single_turn",
    overwrite=True,
)
