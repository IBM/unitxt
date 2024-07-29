from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "question": str,
            "answer_a": str,
            "answer_b": str,
            "model_a": str,
            "model_b": str,
        },
        reference_fields={
            "answer_a_preference": int,  # Positive numbers for preferring answer_a, negative for answer_b.
        },
        prediction_type=int,
        metrics=["metrics.weighted_win_rate_correlation", "metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparative_rating.single_turn",
    overwrite=True,
)
