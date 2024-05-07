from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "question": "str",
            "model_a_answer": "str",
            "model_b_answer": "str",
            "reference_answer": "str",
            "model_a_label": "str",
            "model_b_label": "str",
            "tie_label": "str",
        },
        outputs={"winner": "str"},
        metrics=["metrics.accuracy"],
    ),
    "tasks.model_response_assessment.model_pairwise_comparison_with_reference_single_turn",
    overwrite=True,
)
