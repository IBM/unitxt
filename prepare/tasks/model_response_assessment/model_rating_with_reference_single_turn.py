from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"question": "str", "model_answer": "str", "reference_answer": "str"},
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.model_response_assessment.model_rating_with_reference_single_turn",
    overwrite=True,
)
