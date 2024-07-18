from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "dialog": "List[Tuple[str, str]]",
            "reference_dialog": "List[Tuple[str, str]]",
        },
        reference_fields={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.response_assessment.rating.multi_turn_with_reference",
    overwrite=True,
)
