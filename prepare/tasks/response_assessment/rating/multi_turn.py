from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"dialog": "List[Tuple[str, str]]"},
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.response_assessment.rating.multi_turn",
    overwrite=True,
)
