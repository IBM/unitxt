from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "model_a_dialog": "List[Tuple[str, str]]",
            "model_b_dialog": "List[Tuple[str, str]]",
            "model_a_label": "str",
            "model_b_label": "str",
            "tie_label": "str",
        },
        outputs={"winner": "str"},
        metrics=["metrics.accuracy"],
    ),
    "tasks.model_response_assessment.model_pairwise_comparison_multi_turn",
    overwrite=True,
)
