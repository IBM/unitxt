from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs=["question", "model_output"],
        outputs=["rating_label"],
        metrics=["metrics.spearman"],
    ),
    "tasks.rag.model_response_assessment",
    overwrite=True,
)
