from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={"table": "str", "output": "str", "reference_output": "str"},
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.response_assessment.rating.table2text_single_turn_with_reference",
    overwrite=True,
)
