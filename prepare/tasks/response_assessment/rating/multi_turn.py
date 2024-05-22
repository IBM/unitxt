from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"dialog": "List[Tuple[str, str]]"},
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.response_assessment.rating.multi_turn",
    overwrite=True,
)
