from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "dialog": "List[Tuple[str, str]]",
            "reference_dialog": "List[Tuple[str, str]]",
        },
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.model_response_assessment.model_rating_with_reference_multi_turn",
    overwrite=True,
)
