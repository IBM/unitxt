from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"question": "str", "answer": "str"},
        outputs={"rating": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.response_assessment.rating.single_turn",
    overwrite=True,
)
