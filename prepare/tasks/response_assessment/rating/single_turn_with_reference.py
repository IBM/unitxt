from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"question": str, "answer": str, "reference_answer": str},
        reference_fields={"rating": float},
        metrics=["metrics.spearman"],
        prediction_type=float,
    ),
    "tasks.response_assessment.rating.single_turn_with_reference",
    overwrite=True,
)
