from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"dialog": "List[Tuple[str, str]]", "reference_answer": "str"},
        outputs={"rating_label": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.model_response_assessment.absolute_score_multi_turn_with_reference",
    overwrite=True,
)
